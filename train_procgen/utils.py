import tensorflow as tf

REWARD_RANGE_FOR_C51 = {
    'starpilot': (0, 35),
    'caveflyer': (0, 13.4),
    'dodgeball': (0, 19),
    'fruitbot': (-5, 27.2),
    'jumper': (0, 10),
    'climber': (0, 12.6)
}

def reduce_std(input_tensor, axis=None, keepdims=False):
    """
    Tensorflow 1.12 compatiable
    """
    means = tf.math.reduce_mean(input_tensor, axis=axis, keepdims=True)
    squared_deviations = tf.math.square(input_tensor - means)
    variance = tf.math.reduce_mean(squared_deviations, axis=axis, keepdims=keepdims)
    return tf.math.sqrt(variance)
