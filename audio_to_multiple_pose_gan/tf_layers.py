import tensorflow as tf

def to_motion_delta(pose_batch):
    shape = pose_batch.get_shape()
    reshaped = tf.reshape(pose_batch, (-1, 64, 2, shape[-1]/2))
    diff = reshaped[:, 1:] - reshaped[:, :-1]
    return tf.reshape(diff, (-1, 63, shape[-1]))


def keypoints_to_train(poses, arr):
    shape = poses.get_shape()
    reshaped = tf.reshape(poses, (tf.shape(poses)[0], shape[1].value, 2, 49))
    required_keypoints = tf.gather(reshaped, indices=arr, axis=3)
    return tf.reshape(required_keypoints, (tf.shape(poses)[0], shape[1].value, 2*len(arr)))


def keypoints_regloss(gt_keypoints, pred_keypoints, regloss_type):
    if regloss_type == 'l1':
        loss_func = tf.keras.losses.mean_absolute_error
    elif regloss_type == 'l2':
        loss_func = tf.keras.losses.mean_squared_error
    else:
        raise ValueError("Wrong regression loss")
    return tf.reduce_mean(loss_func(tf.layers.flatten(gt_keypoints), tf.layers.flatten(pred_keypoints)))


def GroupNorm(x, type='1d', G=32, esp=1e-5):
    #https://github.com/shaohua0116/Group-Normalization-Tensorflow/blob/551511c939fc5733a61e2505cc03774a6d224547/ops.py#L15
    if type == '1d':
        # tranpose: [bs, l, c] to [bs, c, l] following the paper
        x = tf.transpose(x, [0, 2, 1])
        N, C, L = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, L])
        mean, var = tf.nn.moments(x, [2, 3], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
        beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
        gamma = tf.reshape(gamma, [1, C, 1])
        beta = tf.reshape(beta, [1, C, 1])

        output = tf.reshape(x, [-1, C, L]) * gamma + beta
        # tranpose: [bs, c, l] to [bs, l, c] following the paper
        output = tf.transpose(output, [0, 2, 1])
    elif type == '2d':
        # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
        beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
    else:
        raise ValueError('Unimplemented GroupNorm type.')
    return output


def Norm(input, norm='batch', type='1d', is_training=True):
    if norm == 'batch':
        output = tf.layers.batch_normalization(input, training=is_training)
    elif norm == 'instance':
        output = tf.contrib.layers.instance_norm(input, trainable=False)
    elif norm == 'group':
        if type == '1d':
            output = GroupNorm(input, type='1d')
            #output = tf.contrib.layers.group_norm(input, reduction_axes=(-2,), trainable=False)
        elif type == '2d':
            output = GroupNorm(input, type='2d')
            #output = tf.contrib.layers.group_norm(input, reduction_axes=(-3, -2), trainable=False)
        else:
            raise ValueError('Unimplemented GroupNorm type.')
    else:
        raise ValueError('Unimplemented GroupNorm type.')
    return output


def ConvNormRelu(input, channels, type='1d', is_training=True, leaky=False, downsample=False, norm='batch', k=None,
                 s=None, padding='same'):
    if k == None and s == None:
        if not downsample:
            k = 3
            s = 1
        else:
            k = 4
            s = 2

    if type == '1d':
        conv = tf.layers.conv1d(input, filters=channels, kernel_size=k, strides=s, padding=padding,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), activation=None)
    elif type == '2d':
        conv = tf.layers.conv2d(input, filters=channels, kernel_size=k, strides=s, padding=padding,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                bias_initializer=tf.zeros_initializer(), activation=None)
    else:
        raise ValueError('Unimplemented conv type!!!')
    # add a norm layer
    conv = Norm(conv, norm=norm, type=type, is_training=is_training)

    if leaky:
        return tf.nn.leaky_relu(conv, alpha=0.2)
    return tf.nn.relu(conv)


def UpSampling1D(input):
    return tf.keras.backend.repeat_elements(input, 2, axis=1)