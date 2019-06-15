import numpy as np

class MySimpleScaler(object):
  def __init__(self):
    self._means = None
    self._stds = None

  def preprocess(self, data):
    return data

    # if self._means is None: # during training only
    #   self._means = np.mean(data, axis=0)

    # if self._stds is None: # during training only
    #   self._stds = np.std(data, axis=0)
    #   if not self._stds.all():
    #     raise ValueError('At least one column has standard deviation of 0.')

    # return (data - self._means) / self._stds