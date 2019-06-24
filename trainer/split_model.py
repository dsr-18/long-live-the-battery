import trainer.constants as cst

from tensorflow.keras.layers import concatenate, LSTM, Conv1D, Flatten, TimeDistributed, Input, Dense, MaxPooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf


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


def create_keras_model(window_size, loss, hparams_config=None):
    """Creates the Keras model.
    
    Arguments
    window_size: Number of samples per row. Must match window_size 
    of the datasets that are used to fit/predict.
    
    loss: Loss function of the model.
    
    hparams_config: A dictionary of hyperparameters that can be used
    to test multiple configurations (hpo). Default is the 
    'hparams' dictionary that is defined at the beginning of this
    function. This dictionary is used for standard non-hpo jobs
    and for any parameter that is not defined in a hpo job.
    """
    
    # Default configuration
    hparams = {
        cst.CONV_FILTERS: 8,
        cst.CONV_KERNEL: 9,
        cst.CONV_ACTIVATION: "relu",
        cst.LSTM_NUM_UNITS: 128,
        cst.LSTM_ACTIVATION: 'tanh',
        cst.DENSE_NUM_UNITS: 32,
        cst.DENSE_ACTIVATION: 'relu',
        cst.OUTPUT_ACTIVATION: 'relu',
        cst.LEARNING_RATE: 0.001,
        cst.DROPOUT_RATE_CNN: 0.3,
        cst.DROPOUT_RATE_LSTM: 0.3,
    }
    # update hyperparameters with arguments from task_hyperparameter.py
    if hparams_config:
        hparams.update(hparams_config)

    # define Inputs
    qdlin_in = Input(shape=(window_size, cst.STEPS, cst.INPUT_DIM), name=cst.QDLIN_NAME)
    tdlin_in = Input(shape=(window_size, cst.STEPS, cst.INPUT_DIM), name=cst.TDLIN_NAME)
    ir_in = Input(shape=(window_size, cst.INPUT_DIM), name=cst.INTERNAL_RESISTANCE_NAME)
    dt_in = Input(shape=(window_size, cst.INPUT_DIM), name=cst.DISCHARGE_TIME_NAME)
    qd_in = Input(shape=(window_size, cst.INPUT_DIM), name=cst.QD_NAME)
    
    # combine all data from detail level
    detail_concat = concatenate([qdlin_in, tdlin_in], axis=3, name='detail_concat')

    # define CNN
    cnn_out = TimeDistributed(Conv1D(filters=hparams[cst.CONV_FILTERS],
                                     kernel_size=hparams[cst.CONV_KERNEL],
                                     strides=hparams[cst.CONV_KERNEL]//3,
                                     activation=hparams[cst.CONV_ACTIVATION],
                                     padding='same'), name='convolution')(detail_concat)
    # Add some maxpools to reduce output size
    cnn_maxpool = TimeDistributed(MaxPooling1D(), name='conv_pool')(cnn_out)
    cnn_out2 = TimeDistributed(Conv1D(filters=hparams[cst.CONV_FILTERS] * 2,
                                      kernel_size=hparams[cst.CONV_KERNEL],
                                      strides=hparams[cst.CONV_KERNEL]//3,
                                      activation=hparams[cst.CONV_ACTIVATION],
                                      padding='same'), name='conv2')(cnn_maxpool)
    cnn_maxpool2 = TimeDistributed(MaxPooling1D(), name='pool2')(cnn_out2)
    cnn_out3 = TimeDistributed(Conv1D(filters=hparams[cst.CONV_FILTERS] * 4,
                                      kernel_size=hparams[cst.CONV_KERNEL],
                                      strides=hparams[cst.CONV_KERNEL]//3,
                                      activation=hparams[cst.CONV_ACTIVATION],
                                      padding='same'), name='conv3')(cnn_maxpool2)
    cnn_maxpool3 = TimeDistributed(MaxPooling1D(), name='pool3')(cnn_out3)
    cnn_flat = TimeDistributed(Flatten(), name='convolution_flat')(cnn_maxpool3)
    drop_out = TimeDistributed(Dropout(rate=hparams[cst.DROPOUT_RATE_CNN]), name='dropout_cnn')(cnn_flat)

    # combine CNN output with all data from summary level
    all_concat = concatenate([drop_out, ir_in, dt_in, qd_in], axis=2, name='all_concat')

    # define LSTM
    lstm_out = LSTM(hparams[cst.LSTM_NUM_UNITS], activation=hparams[cst.LSTM_ACTIVATION], name='recurrent')(all_concat)
    drop_out_2 = Dropout(rate=hparams[cst.DROPOUT_RATE_LSTM], name='dropout_lstm')(lstm_out)
    hidden_dense = Dense(hparams[cst.DENSE_NUM_UNITS], name='hidden', activation=hparams[cst.DENSE_ACTIVATION])(drop_out_2)
    # Relu activation on the last layer for striclty positive outputs
    main_output = Dense(2, name='output', activation=hparams[cst.OUTPUT_ACTIVATION])(hidden_dense)

    model = Model(inputs=[qdlin_in, tdlin_in, ir_in, dt_in, qd_in], outputs=[main_output])
    
    metrics_list = [mae_current_cycle, mae_remaining_cycles]
    
    model.compile(loss=loss, optimizer=Adam(lr=hparams[cst.LEARNING_RATE], clipnorm=1.), metrics=metrics_list)

    return model