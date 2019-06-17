import itertools
import argparse
import os
from absl import logging
import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import trainer.data_pipeline as dp
import trainer.split_model as split_model
import trainer.constants as cst
import trainer.task as task
    
    
def train_and_evaluate(hparams, run_dir):
    model = split_model.create_keras_model(window_size=args.window_size,
                               loss=args.loss,
                               hparams_config=hparams)
    
    # calculate steps_per_epoch_train, steps_per_epoch_test
    steps_per_epoch_train = task.calculate_steps_per_epoch(args.data_dir_train, args.window_size, args.shift, args.stride, args.batch_size)
    steps_per_epoch_validate = task.calculate_steps_per_epoch(args.data_dir_validate, args.window_size, args.shift, args.stride, args.batch_size)
    
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
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=run_dir, histogram_freq=0, write_graph=False),
    ]
    
    history = model.fit(
        dataset_train, 
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch_train,
        validation_data=dataset_validate,
        validation_steps=steps_per_epoch_validate,
        verbose=1,
        callbacks=callbacks,
    )    

    mae_current = min(history.history["val_mae_current_cycle"])
    mae_remaining = min(history.history["val_mae_remaining_cycles"])
    return mae_current, mae_remaining


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mae_current, mae_remaining = train_and_evaluate(hparams, run_dir)
        tf.summary.scalar('current_mae', mae_current, step=1)
        tf.summary.scalar('remaining_mae', mae_remaining, step=1)

  
def get_hyperparameter_grid(hyperparameters):
    keys = [param.name for param in hyperparameters]
    values = [param.domain.values for param in hyperparameters]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

  
def grid_search(args):    
    run_timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.tboard_dir is None:
        tboard_dir = os.path.join(cst.TENSORBOARD_DIR, run_timestr + "_gridsearch")
    else:
        tboard_dir = os.path.join(args.tboard_dir + "_gridsearch")
        
    # pick the parameters to tune
    hyperparameters = [
        hp.HParam('NUM_LSTM_UNITS', hp.Discrete([8])),
        hp.HParam('CONV_KERNEL', hp.Discrete([3, 5])),
        hp.HParam('CONV_FILTERS', hp.Discrete([8, 16])),
        hp.HParam('NUM_DENSE_UNITS', hp.Discrete([32])),
        ]
        
    session_num = 0
    
    for hparams in get_hyperparameter_grid(hyperparameters):
        print("RUN TIMESTR: {}".format(run_timestr))
        run_name = "run-{}_{}".format(session_num, run_timestr)
        print('--- Starting trial: {}'.format(run_name))
        print({h: hparams[h] for h in hparams})
        run(os.path.join(tboard_dir, run_name), hparams)
        session_num += 1
            
            
if __name__ == "__main__":                
    args = task.get_args()
    grid_search(args)
    