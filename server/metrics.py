import tensorflow.keras.backend as K
import tensorflow as tf

Remaining_cycles_scaling_factor = 2159.0  # Hardcoded for easier run on google cloud


def mae_remaining_cycles(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    diff = tf.math.abs(y_true - y_pred)
    return K.mean(diff, axis=0)[1] * Remaining_cycles_scaling_factor  # TODO: scaling factor hard coded for now!


def mae_current_cycle(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    diff = tf.math.abs(y_true - y_pred)
    return K.mean(diff, axis=0)[0] * Remaining_cycles_scaling_factor  # TODO: scaling factor hard coded for now!
