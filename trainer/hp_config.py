import trainer.constants as cst
from tensorboard.plugins.hparams import api as hp

"""
Hyperparameter configurations for every available model.
Comment out any lines you don't want to include in the 
hyperparameter optimization - they will default to the
scalar values defined in the matching model file.

Only hp.Discrete() works with hpo gridsearch,
avoid hp.RealInterval and hp.IntInterval.
"""

split_model_hparams = [
   hp.HParam(cst.CONV_KERNEL, hp.Discrete([9, 27])),
   hp.HParam(cst.CONV_FILTERS, hp.Discrete([8, 32])),
   hp.HParam(cst.LSTM_NUM_UNITS, hp.Discrete([64, 128])),
   # hp.HParam(cst.DENSE_NUM_UNITS, hp.Discrete([32])),
   # hp.HParam(cst.OUTPUT_ACTIVATION, hp.Discrete(['relu', 'sigmoid'])),
   # hp.HParam(cst.LEARNING_RATE, hp.Discrete([0.0001, 0.00001, 0.000005])),
   # hp.HParam(cst.DROPOUT_RATE_CNN, hp.Discrete([0.3, 0.4])),
   # hp.HParam(cst.DROPOUT_RATE_LSTM, hp.Discrete([0.3, 0.4])),
   # hp.HParam(cst.CONV_KERNEL_2D, hp.Discrete([(3, 9)])),
   # hp.HParam(cst.CONV_STRIDE_2D, hp.Discrete([(1, 3)])),
   ]
