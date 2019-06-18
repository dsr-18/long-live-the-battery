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
   hp.HParam(cst.CONV_KERNEL, hp.Discrete([3, 5])),
   hp.HParam(cst.CONV_FILTERS, hp.Discrete([8, 16])),
   # hp.HParam(cst.CONV_ACTIVATION, hp.Discrete(['relu', 'sigmoid'])),
   # hp.HParam(cst.LSTM_NUM_UNITS, hp.Discrete([8, 16])),
   hp.HParam(cst.LSTM_ACTIVATION, hp.Discrete(['sigmoid', 'tanh'])),
   # hp.HParam(cst.DENSE_NUM_UNITS, hp.Discrete([32, 64])),
   # hp.HParam(cst.DENSE_ACTIVATION, hp.Discrete(['relu', 'sigmoid']))
   ]