import tensorflow as tf

def build_impala_cnn(unscaled_images, depths=[16,32,32], dense=tf.layers.dense, use_bn=False, randcnn=False, **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())
        if use_bn:
            # always use train mode BN
            out = tf.layers.batch_normalization(out, center=True, scale=True, training=True)
        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    if randcnn:
        out = tf.layers.conv2d(out, 3, 3, padding='same', 
                               kernel_initializer=tf.initializers.glorot_normal(), 
                               trainable=False, 
                               name='randcnn')

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out