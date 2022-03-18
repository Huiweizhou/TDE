from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inits import zeros, glorot, ones

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

seed = 2021
tf.random.set_seed(seed)
# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.compat.v1.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.compat.v1.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.compat.v1.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False,
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        print(input_dim, output_dim)

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.compat.v1.get_variable('weights', shape=(input_dim, output_dim),
                                                             dtype=tf.float32,
                                                             initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                                                                 scale=1.0, mode="fan_avg", distribution="uniform"),
                                                             # initializer=tf.keras.initializers.GlorotUniform(),
                                                             regularizer=tf.keras.regularizers.l2(
                                                                 0.5 * FLAGS.weight_decay))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - (1 - self.dropout))

        # transform
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class RNN_GRU(Layer):
    """gru layer."""

    def __init__(self, input_dim, output_dim, dropout=0.,
                 placeholders=None, bias=False, featureless=False,
                 sparse_inputs=False, **kwargs):
        super(RNN_GRU, self).__init__(**kwargs)

        self.dropout = dropout

        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        print(input_dim, output_dim)

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['W_r'] = glorot([input_dim, output_dim], name='W_r')
            self.vars['U_r'] = glorot([input_dim, output_dim], name='U_r')
            self.vars['W_z'] = glorot([input_dim, output_dim], name='W_z')
            self.vars['U_z'] = glorot([input_dim, output_dim], name='U_z')
            self.vars['W_h'] = glorot([input_dim, output_dim], name='W_h')
            self.vars['U_h'] = glorot([input_dim, output_dim], name='U_h')
            # self.vars['W_o'] = glorot([input_dim, output_dim], name='W_o')
            if self.bias:
                self.vars['bias_gru'] = zeros([output_dim], name='bias_gru')

        if self.logging:
            self._log_vars()

    def _call(self, inputs, hidden):
        x = inputs
        r_t = tf.sigmoid(tf.matmul(x, self.vars['W_r']) + tf.matmul(hidden, self.vars['U_r']))
        z_t = tf.sigmoid(tf.matmul(x, self.vars['W_z']) + tf.matmul(hidden, self.vars['U_z']))
        h_t_1 = tf.tanh(tf.matmul(x, self.vars['W_h']) + tf.matmul(tf.math.multiply(r_t, hidden), self.vars['U_h']))
        h_t = tf.math.multiply(1 - z_t, hidden) + tf.math.multiply(z_t, h_t_1)
        # y_t = tf.sigmoid(tf.matmul(h_t, self.vars['W_o']))
        return h_t

    def __call__(self, inputs, hidden):
        with tf.compat.v1.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.compat.v1.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs, hidden)
            if self.logging:
                tf.compat.v1.summary.histogram(self.name + '/outputs', outputs)
            return outputs


class T2V(Layer):
    def __init__(self, output_dim=None, input_shape=[6, 1], **kwargs):
        self.output_dim = output_dim - 1
        super(T2V, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['W'] = glorot(name='W', shape=(self.output_dim, self.output_dim))
            self.vars['B'] = glorot(name='B', shape=(input_shape[1], self.output_dim))
            self.vars['w'] = glorot(name='w', shape=(1, 1))
            self.vars['b'] = glorot(name='b', shape=(input_shape[1], 1))

    def _call(self, x):
        original = self.vars['w'] * x + self.vars['b']
        x = tf.compat.v1.keras.backend.repeat_elements(x, self.output_dim, -1)
        # print(x)
        sin_trans = tf.sin(tf.matmul(x, self.vars['W']) + self.vars['B'])
        return tf.concat([sin_trans, original], -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim + 1)


# 实现conv2d
class Conv2d(Layer):
    """Dense layer."""

    def __init__(self, dropout=0., output_dim=256,
                 act=tf.nn.relu, bias=True, featureless=False,
                 sparse_inputs=False, **kwargs):
        super(Conv2d, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.featureless = featureless
        self.bias = bias

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            for i in range(50):
                self.vars['weights' + str(i)] = glorot(shape=[1, 256, 1, 1], name='weights' + str(i))

                if self.bias:
                    self.vars['bias' + str(i)] = zeros([1, 3, 1, 1], name='bias' + str(i))
            for i in range(50,100):
                self.vars['weights' + str(i)] = glorot(shape=[2, 256, 1, 1], name='weights' + str(i))

                if self.bias:
                    self.vars['bias' + str(i)] = zeros([1, 2, 1, 1], name='bias' + str(i))
            for i in range(100,output_dim):
                self.vars['weights' + str(i)] = glorot(shape=[3, 256, 1, 1], name='weights' + str(i))

                if self.bias:
                    self.vars['bias' + str(i)] = zeros([1, 1, 1, 1], name='bias' + str(i))

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        for i in range(50):
            output1 = tf.nn.conv2d(x, self.vars['weights' + str(i)], strides=[1, 1, 1, 1], padding='VALID')
            if self.bias:
                output1 = tf.nn.bias_add(output1, self.vars['bias' + str(i)])
            output1 = tf.nn.max_pool2d(output1, ksize=(1, 3, 1, 1), strides=(1, 1, 1, 1), padding='VALID')
            if i==0:
                output = output1
            else:
                output = tf.concat((output, output1), axis=1)
        for i in range(50, 100):
            output1 = tf.nn.conv2d(x, self.vars['weights' + str(i)], strides=[1, 1, 1, 1], padding='VALID')
            if self.bias:
                output1 = tf.nn.bias_add(output1, self.vars['bias' + str(i)])
            output1 = tf.nn.max_pool2d(output1, ksize=(1, 2, 1, 1), strides=(1, 1, 1, 1), padding='VALID')
            output = tf.concat((output, output1), axis=1)
        for i in range(100, 256):
            output1 = tf.nn.conv2d(x, self.vars['weights' + str(i)], strides=[1, 1, 1, 1], padding='VALID')
            if self.bias:
                output1 = tf.nn.bias_add(output1, self.vars['bias' + str(i)])
            output = tf.concat((output, output1), axis=1)

        return output
