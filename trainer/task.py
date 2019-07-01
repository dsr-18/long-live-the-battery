import argparse
import datetime
import os

import tensorflow as tf
from absl import logging

import trainer.constants as cst
import trainer.data_pipeline as dp
import trainer.split_model as split_model
import trainer.full_cnn_model as full_cnn_model
from trainer.callbacks import CustomCheckpoints


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
        default=32,
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
    parser.add_argument(
        '--model',
        default='split_model',
        type=str,
        help='The type of model to use, default="split_model", options="split_model", "full_cnn_model"'
    )
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args, tboard_dir, hparams=None):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in data_pipeline.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
    args: dictionary of arguments - see get_args() for details
    """
    # Config datasets for consistent usage
    ds_config = dict(window_size=args.window_size,
                     shift=args.shift,
                     stride=args.stride,
                     batch_size=args.batch_size)
    ds_train_path = args.data_dir_train
    ds_val_path = args.data_dir_validate

    # create model
    if args.model == 'split_model':
        print("Using split model!")
        model = split_model.create_keras_model(window_size=ds_config["window_size"],
                                               loss=args.loss,
                                               hparams_config=hparams)
    if args.model == 'full_cnn_model':
        print("Using full cnn model!")
        model = full_cnn_model.create_keras_model(window_size=ds_config["window_size"],
                                                  loss=args.loss,
                                                  hparams_config=hparams)
    
    # Calculate steps_per_epoch_train, steps_per_epoch_test
    # This is needed, since for counting repeat has to be false
    steps_per_epoch_train = calculate_steps_per_epoch(data_dir=ds_train_path, dataset_config=ds_config)
    
    steps_per_epoch_validate = calculate_steps_per_epoch(data_dir=ds_val_path, dataset_config=ds_config)
    
    # load datasets
    dataset_train = dp.create_dataset(data_dir=ds_train_path,
                                      window_size=ds_config["window_size"],
                                      shift=ds_config["shift"],
                                      stride=ds_config["stride"],
                                      batch_size=ds_config["batch_size"])
    
    dataset_validate = dp.create_dataset(data_dir=ds_val_path,
                                         window_size=ds_config["window_size"],
                                         shift=ds_config["shift"],
                                         stride=ds_config["stride"],
                                         batch_size=ds_config["batch_size"])
    
    # if hparams is passed, we're running a HPO-job
    if hparams:
        checkpoint_callback = CustomCheckpoints(save_last_only=True,
                                                log_dir=tboard_dir,
                                                dataset_path=ds_val_path,
                                                dataset_config=ds_config,
                                                save_eval_plot=False)
    else:
        checkpoint_callback = CustomCheckpoints(save_best_only=True,
                                                start_epoch=args.save_from,
                                                log_dir=tboard_dir,
                                                dataset_path=ds_val_path,
                                                dataset_config=ds_config,
                                                save_eval_plot=False)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tboard_dir,
                                       histogram_freq=0,
                                       write_graph=False,
                                       ),
        checkpoint_callback,
    ]

    model.summary()
    
    # train model
    history = model.fit(
        dataset_train, 
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch_train,
        validation_data=dataset_validate,
        validation_steps=steps_per_epoch_validate,
        verbose=2,
        callbacks=callbacks)
    
    mae_current = min(history.history["val_mae_current_cycle"])
    mae_remaining = min(history.history["val_mae_remaining_cycles"])
    return mae_current, mae_remaining


def calculate_steps_per_epoch(data_dir, dataset_config):
    temp_dataset = dp.create_dataset(data_dir=data_dir,
                                     window_size=dataset_config["window_size"],
                                     shift=dataset_config["shift"],
                                     stride=dataset_config["stride"],
                                     batch_size=dataset_config["batch_size"],
                                     repeat=False)
    steps_per_epoch = 0
    for batch in temp_dataset:
        steps_per_epoch += 1
    return steps_per_epoch


def get_tboard_dir():
    run_timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.tboard_dir is None:
        tboard_dir = os.path.join(cst.TENSORBOARD_DIR, "jobs", run_timestr)
    else:
        tboard_dir = args.tboard_dir
    return tboard_dir


if __name__ == '__main__':
    args = get_args()
    logging.set_verbosity(args.verbosity)
    train_and_evaluate(args, get_tboard_dir())
