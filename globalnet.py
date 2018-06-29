import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import conv2d,utils

channel_stride = [(256, 64, 1)] * 3 +\
                 [(512, 128, 2)] + [(512, 128, 1)] * 3+\
                 [(1024, 256, 2)] + [(1024, 256, 1)] * 5 +\
                 [(2048, 512, 2)] + [(2048, 512, 1)] * 2
conv2d(inputs=inp,
       num_outputs=,
       kernel_size=,
       stride=1,
       padding='SAME',
       data_format=None,
       rate=1,
       activation_fn=nn.relu,
       normalizer_fn=None,
       normalizer_params=None,
       weights_initializer=initializers.xavier_initializer(),
       weights_regularizer=None,
       biases_initializer=init_ops.zeros_initializer(),
       biases_regularizer=None,
       reuse=None,
       variables_collections=None,
       outputs_collections=None,
       trainable=True,
       scope=None)


def bottleneck(inputs,block_arg):
    narrow_width, wide_width, block_stride=block_arg


# dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
# dataset = tf.data.Dataset.range(max_value)
# iter=dataset.make_one_shot_iterator()
# print(type(iter))
# aa=iter.get_next()
# print(type(aa))
# a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
# b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
# c = a + b
# sess= tf.Session(config=tf.ConfigProto(log_device_placement=False))
# print(sess.run(c))

