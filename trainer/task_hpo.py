import itertools
import os
import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import trainer.data_pipeline as dp
import trainer.split_model as split_model
import trainer.constants as cst
import trainer.task as task
from trainer.hp_config import split_model_hparams
from trainer.callbacks import CustomCheckpoints
    
    
def train_and_evaluate(hparams, run_dir, args):
    model = split_model.create_keras_model(window_size=args.window_size,
                                           loss=args.loss,
                                           hparams_config=hparams)
    
    # Config datasets for consistent usage
    ds_config = dict(window_size=args.window_size,
                     shift=args.shift,
                     stride=args.stride,
                     batch_size=args.batch_size)
    ds_train_path = args.data_dir_train
    ds_val_path = args.data_dir_validate

    # Calculate steps_per_epoch_train, steps_per_epoch_test
    # This is needed, since for counting repeat has to be false
    steps_per_epoch_train = task.calculate_steps_per_epoch(data_dir=ds_train_path, dataset_config=ds_config)
    
    steps_per_epoch_validate = task.calculate_steps_per_epoch(data_dir=ds_val_path, dataset_config=ds_config)
    
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
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=run_dir, histogram_freq=0, write_graph=False),
        CustomCheckpoints(log_dir=run_dir,
                          save_last_only=True,
                          start_epoch=args.save_from,
                          dataset_path=ds_val_path,
                          dataset_config=ds_config)
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


def run(run_dir, hparams, args):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        mae_current, mae_remaining = train_and_evaluate(hparams, run_dir, args)
        tf.summary.scalar('current_mae', mae_current, step=1)
        tf.summary.scalar('remaining_mae', mae_remaining, step=1)
        
        
def get_hyperparameter_grid(hyperparameters):
    keys = [param.name for param in hyperparameters]
    values = [param.domain.values for param in hyperparameters]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

  
def grid_search(args):    
    run_timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.tboard_dir is None:
        tboard_dir = os.path.join(cst.TENSORBOARD_DIR, "gridsearches", run_timestr + "_gridsearch")
    else:
        tboard_dir = os.path.join(args.tboard_dir + "_gridsearch")
        
    # to pick parameters that are iterated over, edit hp_config.py
    hyperparameters = split_model_hparams
    
    session_num = 0
    for hparams in get_hyperparameter_grid(hyperparameters):
        print("RUN TIMESTR: {}".format(run_timestr))
        run_name = "run-{}_{}".format(session_num, run_timestr)
        print('--- Starting trial: {}'.format(run_name))
        print({h: hparams[h] for h in hparams})
        run(os.path.join(tboard_dir, run_name), hparams, args)
        session_num += 1
            
            
if __name__ == "__main__":                
    args = task.get_args()
    grid_search(args)
    