import argparse

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import trainer.data_pipeline as dp
import trainer.split_model as split_model
import trainer.constants as cst

from tensorflow.keras.layers import concatenate, LSTM, Conv1D, Flatten, TimeDistributed, Input, Dense, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf


def get_args():
    """Argument parser.

    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        default=cst.BASE_DIR,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--data-dir-train',
        type=str,
        default=cst.TRAIN_SET,
        help='local or GCS location for reading TFRecord files for the training set')
    parser.add_argument(
        '--data-dir-validate',
        type=str,
        default=cst.TEST_SET,
        help='local or GCS location for reading TFRecord files for the validation set')
    parser.add_argument(
        '--tboard-dir',         # no default so we can construct dynamically with timestamp
        type=str,
        help='local or GCS location for reading TensorBoard files')
    parser.add_argument(
        '--saved-model-dir',    # no default so we can construct dynamically with timestamp
        type=str,
        help='local or GCS location for saving trained Keras models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='number of times to go through the data, default=3')
    parser.add_argument(
        '--batch-size',
        default=16,
        type=int,
        help='number of records to read during each training step, default=16')
    parser.add_argument(
        '--window-size',
        default=20,
        type=int,
        help='window size for sliding window in training sample generation, default=100')
    parser.add_argument(
        '--shift',
        default=5,
        type=int,
        help='shift for sliding window in training sample generation, default=20')
    parser.add_argument(
        '--stride',
        default=1,
        type=int,
        help='stride inside sliding window in training sample generation, default=1')
    parser.add_argument(
        '--learning-rate',
        default=.01,      # NOT USED RIGHT NOW
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='DEBUG')
    parser.add_argument(
        '--loss',
        default='mean_squared_error',
        type=str,
        help='loss function used by the model, default=mean_squared_error')
    parser.add_argument(
        '--shuffle',
        default=True,
        type=bool,
        help='shuffle the batched dataset, default=True'
    )
    parser.add_argument(
        '--shuffle-buffer',
        default=500,
        type=int,
        help='Bigger buffer size means better shuffling but longer setup time. Default=500'
    )
    parser.add_argument(
        '--save-from',
        default=80,
        type=int,
        help='epoch after which model checkpoints are saved, default=80'
    )
    args, _ = parser.parse_known_args()
    return args


def calculate_steps_per_epoch(data_dir, window_size, shift, stride, batch_size):
    temp_dataset = dp.create_dataset(
                        data_dir=data_dir,
                        window_size=window_size,
                        shift=shift,
                        stride=stride,
                        batch_size=batch_size,
                        repeat=False)
    steps_per_epoch = 0
    for batch in temp_dataset:
        steps_per_epoch += 1
    return steps_per_epoch


def mae_remaining_cycles(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    diff = tf.math.abs(y_true - y_pred)
    return K.mean(diff, axis=0)[1] * 2159.0  # TODO: scaling factor hard coded for now!


def mae_current_cycle(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    diff = tf.math.abs(y_true - y_pred)
    return K.mean(diff, axis=0)[0] * 2159.0  # TODO: scaling factor hard coded for now!


def mape_current_cycle(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    diff = tf.math.abs(
        (y_true - y_pred) / K.clip(tf.math.abs(y_true), 1 / 2159.0, None))
    return 100. * K.mean(diff, axis=0)[0]

    
def mape_remaining_cycles(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    diff = tf.math.abs(
        (y_true - y_pred) / K.clip(tf.math.abs(y_true), 1 / 2159.0, None))
    return 100. * K.mean(diff, axis=0)[1]

    
def log_acc_ratio_current_cycle(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    y_true_clipped = K.clip(tf.math.abs(y_true), 1 / 2159.0, None)
    log_acc_ratio = tf.math.log((tf.math.abs(y_pred) / y_true_clipped) + 1)
    return K.mean(log_acc_ratio, axis=0)[0]

    
def log_acc_ratio_remaining_cycles(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    y_true_clipped = K.clip(tf.math.abs(y_true), 1 / 2159.0, None)
    log_acc_ratio = tf.math.log((tf.math.abs(y_pred) / y_true_clipped) + 1)
    return K.mean(log_acc_ratio, axis=0)[1]

    
def log_acc_ratio_loss(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    y_true_clipped = K.clip(tf.math.abs(y_true), 1 / 2159.0, None)
    log_acc_ratio = tf.math.log((tf.math.abs(y_pred) / y_true_clipped) + 1)
    return tf.math.reduce_mean(log_acc_ratio)


# Define hyperparameters
HP_NUM_LSTM_UNITS = hp.HParam('lstm_units', hp.Discrete([16, 32]))
HP_CONV_FILTERS = hp.HParam('conv1d_filters', hp.Discrete([8, 16]))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_CURRENT_MAE = 'current_mae'
METRIC_REMAINING_MAE = 'remaining_mae'
    

def create_keras_model(window_size, loss, hparams=None,):
    """Creates the Keras model.

    Args:
    window_size: [...]
    """
    if hparams == None:
        hparams[HP_NUM_LSTM_UNITS] = 64
        hparams[HP_CONV_FILTERS] = 16
    # define Inputs
    qdlin_in = Input(shape=(window_size, cst.STEPS, cst.INPUT_DIM), name=cst.QDLIN_NAME)
    tdlin_in = Input(shape=(window_size, cst.STEPS, cst.INPUT_DIM), name=cst.TDLIN_NAME)
    ir_in = Input(shape=(window_size, cst.INPUT_DIM), name=cst.INTERNAL_RESISTANCE_NAME)
    dt_in = Input(shape=(window_size, cst.INPUT_DIM), name=cst.DISCHARGE_TIME_NAME)
    qd_in = Input(shape=(window_size, cst.INPUT_DIM), name=cst.QD_NAME)
    
    # combine all data from detail level
    detail_concat = concatenate([qdlin_in, tdlin_in], axis=3, name='detail_concat')

    # define CNN
    cnn_out = TimeDistributed(Conv1D(filters=hparams[HP_CONV_FILTERS], kernel_size=5, activation='relu'), name='convolution')(detail_concat)
    # Add some maxpools to reduce output size
    cnn_maxpool = TimeDistributed(MaxPooling1D(), name='conv_pool')(cnn_out)
    cnn_out2 = TimeDistributed(Conv1D(filters=hparams[HP_CONV_FILTERS], kernel_size=5, activation='relu'), name='conv2')(cnn_maxpool)
    cnn_maxpool2 = TimeDistributed(MaxPooling1D(), name='pool2')(cnn_out2)
    cnn_flat = TimeDistributed(Flatten(), name='convolution_flat')(cnn_maxpool2)

    # combine CNN output with all data from summary level
    all_concat = concatenate([cnn_flat, ir_in, dt_in, qd_in], axis=2, name='all_concat')

    # define LSTM
    lstm_out = LSTM(hparams[HP_NUM_LSTM_UNITS], activation='relu', name='recurrent')(all_concat)
    hidden_dense = Dense(32, name='hidden', activation='relu')(lstm_out)
    main_output = Dense(2, name='output')(hidden_dense)  # Try different activations that are not negative

    model = Model(inputs=[qdlin_in, tdlin_in, ir_in, dt_in, qd_in], outputs=[main_output])
    
    metrics_list = [mae_current_cycle, mae_remaining_cycles]
    
    model.compile(loss=loss, optimizer=Adam(clipnorm=1.), metrics=metrics_list)  # Try lower learning rate

    return model


def train_test_model(hparams):
    model = create_keras_model(window_size=args.window_size,
                               loss=args.loss,
                               hparams=hparams)
    
    # calculate steps_per_epoch_train, steps_per_epoch_test
    steps_per_epoch_train = calculate_steps_per_epoch(args.data_dir_train, args.window_size, args.shift, args.stride, args.batch_size)
    steps_per_epoch_validate = calculate_steps_per_epoch(args.data_dir_validate, args.window_size, args.shift, args.stride, args.batch_size)
    
    # load datasets
    dataset_train = dp.create_dataset(
                        data_dir=args.data_dir_train,
                        window_size=args.window_size,
                        shift=args.shift,
                        stride=args.stride,
                        batch_size=args.batch_size)

    dataset_validate = dp.create_dataset(
                        data_dir=args.data_dir_validate,
                        window_size=args.window_size,
                        shift=args.shift,
                        stride=args.stride,
                        batch_size=args.batch_size)
    
    model.fit(
        dataset_train, 
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch_train,
        #validation_data=dataset_validate,
        #validation_steps=steps_per_epoch_validate,
        verbose=1,
    )    
    
    _, mae_current, mae_remaining = model.evaluate(dataset_validate,
                                 steps=steps_per_epoch_validate)
    return mae_current, mae_remaining


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mae_current, mae_remaining = train_test_model(hparams)
        tf.summary.scalar(METRIC_CURRENT_MAE, mae_current, step=1)
        tf.summary.scalar(METRIC_REMAINING_MAE, mae_remaining, step=1)

  
if __name__ == "__main__":                
    args = get_args()
    session_num = 0
    for num_units in HP_NUM_LSTM_UNITS.domain.values:
        for filters in HP_CONV_FILTERS.domain.values:
        #     for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_LSTM_UNITS: num_units,
                HP_CONV_FILTERS: filters,
                # HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1
