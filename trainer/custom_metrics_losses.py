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


def mape_current_cycle(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    diff = tf.math.abs(
        (y_true - y_pred) / K.clip(tf.math.abs(y_true), 1 / Remaining_cycles_scaling_factor, None))
    return 100. * K.mean(diff, axis=0)[0]

    
def mape_remaining_cycles(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    diff = tf.math.abs(
        (y_true - y_pred) / K.clip(tf.math.abs(y_true), 1 / Remaining_cycles_scaling_factor, None))
    return 100. * K.mean(diff, axis=0)[1]

    
def log_acc_ratio_current_cycle(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    y_true_clipped = K.clip(tf.math.abs(y_true), 1 / Remaining_cycles_scaling_factor, None)
    log_acc_ratio = tf.math.log((tf.math.abs(y_pred) / y_true_clipped) + 1)
    return K.mean(log_acc_ratio, axis=0)[0]

    
def log_acc_ratio_remaining_cycles(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    y_true_clipped = K.clip(tf.math.abs(y_true), 1 / Remaining_cycles_scaling_factor, None)
    log_acc_ratio = tf.math.log((tf.math.abs(y_pred) / y_true_clipped) + 1)
    return K.mean(log_acc_ratio, axis=0)[1]

    
def log_acc_ratio_loss(y_true, y_pred):
    """Copied from tf.keras.losses.mean_absolute_percentage_error and changed the axis to 0.
    This calculates the mean over the two different outputs of the model."""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    y_true_clipped = K.clip(tf.math.abs(y_true), 1 / Remaining_cycles_scaling_factor, None)
    log_acc_ratio = tf.math.log((tf.math.abs(y_pred) / y_true_clipped) + 1)
    return tf.math.reduce_mean(log_acc_ratio)