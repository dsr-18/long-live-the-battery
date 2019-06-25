# Redundant impl of custom activation copied from trainer.split_model
# Copied here to keep server independently runnable
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

def clipped_relu(x):
    return K.relu(x, max_value=1.2)

class Clippy(Activation):
    def __init__(self, activation, **kwargs):
        super(Clippy, self).__init__(activation, **kwargs)
        self.__name__ = 'clippy'
