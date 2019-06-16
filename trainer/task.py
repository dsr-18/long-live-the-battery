import argparse
import os
from absl import logging
import datetime
import numpy as np

import tensorflow as tf

import trainer.data_pipeline as dp
import trainer.split_model as split_model
import trainer.constants as cst


class CustomCheckpoints(tf.keras.callbacks.Callback):
    
    def __init__(self, log_dir, save_best_only=False, period=1, start_epoch=0):
        self.log_dir = log_dir
        self.save_best_only = save_best_only
        self.period = period
        self.start_epoch = start_epoch
        
    def on_train_begin(self, logs=None):
        self.last_saved_epoch = 0
        self.lowest_loss = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.period == 0 and epoch >= self.start_epoch):
            self.current_loss = logs.get('val_loss')
            self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints", "epoch_{}_loss_{}".format(epoch, self.current_loss))
            if self.save_best_only:
                if self.current_loss < self.lowest_loss:
                    tf.keras.experimental.export_saved_model(self.model, self.checkpoint_dir)
                    self.lowest_loss = self.current_loss
                    self.last_saved_epoch = epoch
            else:
                tf.keras.experimental.export_saved_model(self.model, self.checkpoint_dir)

    def on_train_end(self, logs=None):
        print("Last saved model is from epoch {}".format(self.last_saved_epoch))
        
        
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
        help='Bigger buffer size means better shuffling but longer setup time.'
    )
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in data_pipeline.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
    args: dictionary of arguments - see get_args() for details
    """

    # calculate steps_per_epoch_train, steps_per_epoch_test
    steps_per_epoch_train = 10#calculate_steps_per_epoch(args.data_dir_train, args.window_size, args.shift, args.stride, args.batch_size)
    steps_per_epoch_validate = 10#calculate_steps_per_epoch(args.data_dir_validate, args.window_size, args.shift, args.stride, args.batch_size)
    
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

    # create model
    model = split_model.create_keras_model(window_size=args.window_size,
                                           loss=args.loss)

    run_timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.tboard_dir is None:
        tboard_dir = os.path.join(cst.TENSORBOARD_DIR, run_timestr)
    else:
        tboard_dir = args.tboard_dir

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tboard_dir,
                                       write_graph=True,
                                       histogram_freq=0,
                                       write_images=True
                                       ),
        CustomCheckpoints(log_dir=tboard_dir,
                          save_best_only=True,
                          start_epoch=80,
        )
        ]

    # train model
    model.fit(
        dataset_train, 
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch_train,
        validation_data=dataset_validate,
        validation_steps=steps_per_epoch_validate,
        verbose=1,
        callbacks=callbacks)
    
    # export saved model
    if args.saved_model_dir is None:
        saved_model_dir = os.path.join(cst.SAVED_MODELS_DIR_LOCAL, run_timestr)
    else:
        saved_model_dir = args.saved_model_dir
    tf.keras.experimental.export_saved_model(model, saved_model_dir)


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


if __name__ == '__main__':
    args = get_args()
    logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)