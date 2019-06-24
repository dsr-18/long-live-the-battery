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
   hp.HParam(cst.CONV_KERNEL, hp.Discrete([7, 13])),
   # hp.HParam(cst.CONV_FILTERS, hp.Discrete([16, 32])),
   # hp.HParam(cst.CONV_ACTIVATION, hp.Discrete(['relu', 'sigmoid'])),
   # hp.HParam(cst.CONV_STRIDE, hp.Discrete([1,3])),
   hp.HParam(cst.LSTM_NUM_UNITS, hp.Discrete([128, 64])),
   # hp.HParam(cst.LSTM_ACTIVATION, hp.Discrete(['sigmoid', 'tanh'])),
   # hp.HParam(cst.DENSE_NUM_UNITS, hp.Discrete([32, 64])),
   # hp.HParam(cst.DENSE_ACTIVATION, hp.Discrete(['relu', 'sigmoid']))
   # hp.HParam(cst.OUTPUT_ACTIVATION, hp.Discrete(['relu', 'sigmoid'])),
   # hp.HParam(cst.LEARNING_RATE, hp.Discrete([0.0001, 0.00001, 0.000005])),
   # hp.HParam(cst.DROPOUT_RATE_CNN, hp.Discrete([0.3, 0.4])),
   # hp.HParam(cst.DROPOUT_RATE_LSTM, hp.Discrete([0.3, 0.4])),
   ]
