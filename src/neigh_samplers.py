from __future__ import division
from __future__ import print_function

import time
from layers import Layer
import tensorflow as tf
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
tf.compat.v1.disable_eager_execution()

seed = 2021
tf.random.set_seed(seed)
"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """

    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):

        ids, num_samples, t = inputs  # Tensor("h_t_transposed/while/TensorArrayReadV3:0", shape=(), dtype=int32)
        with tf.device("/cpu:0"):
            adj_lists = tf.gather(params=self.adj_info[t], indices=tf.reshape(ids, [
                -1]))  # 'h_t_transposed/while/uniformneighborsampler_1/Slice:0'
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])
        return adj_lists

