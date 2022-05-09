from __future__ import division
from __future__ import print_function

from networkx.generators.random_graphs import fast_gnp_random_graph

import tensorflow as tf
import os
# tf.compat.v1.set_random_seed(123)
# import tensorflow_addons as tfa
import models_util as models
import layers as layers
import tensorflow.keras as tfk

import numpy as np
from aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

tf.compat.v1.disable_eager_execution()

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

seed=2021
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)
class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, pairs, labels, num_window,
                 placeholders, p2v,features, context_features, seq,  # degrees,
                 layer_infos, concat=True, aggregator_type="mean",
                 model_size="small", sigmoid_loss=False, identity_dim=0,
                 **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - neg_adj: Numpy array with negative adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.is_train = placeholders["is_train"]
        # self.is_train = False
        self.model_size = model_size
        self.pairs = pairs
        self.seq = seq
        self.prior = 0.5
        self.nnpu = True
        self.beta = 0
        self.gamma = 1
        if identity_dim > 0:
            self.embeds = tf.compat.v1.get_variable("node_embeddings",
                                                    [features.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features_plhdr = tf.compat.v1.placeholder(dtype=tf.float32, shape=features.shape)
            self.features = tf.compat.v1.get_variable('feature', features.shape, trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)

        self.dtype = tf.float32
        self.concat = concat
        self.num_window = num_window
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.latent_dim = 64
        self.hidden_size = 128
        self.range = tf.Variable(tf.range(0, num_window, 1, dtype=tf.int32), trainable=False)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.lambda_ = 0.0000001

        self.build()
        dummy_emb = tf.zeros_like(tf.expand_dims(self.inputs1, -1), tf.float32)
        h_0 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(1, self.dim_mult * self.dims[-1])),
                        name='h_0')
        tmp_h_0 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(1, self.dim_mult * self.dims[-1])),
                        name='tmp_h_0')
        losses = tf.zeros_like(0.0)
        self.h_t, self.tmp_h_0 , self.losses = tf.scan(self.forward, self.range, initializer=[h_0, tmp_h_0, losses], parallel_iterations=1,
                                        name='h_t_transposed', swap_memory=True)

        self.loss = tf.reduce_mean(self.losses)
        grads_and_vars = self.optimizer.compute_gradients(loss=self.loss)

        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.histogram('h_t', self.h_t[-1, :, :])
        self.merged = tf.compat.v1.summary.merge_all()

        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.pre_preds=self.h_t
        self.preds = tf.nn.sigmoid(self.node_pred(self.h_t[-1, :, :]))

        flattened_emb = tf.reshape(self.h_t, [-1, self.dim_mult * self.dims[-1]])
        self.all_preds = tf.reshape(tf.nn.sigmoid(self.node_pred(flattened_emb)), [num_window, -1, 1])


    def build(self):
        with tf.compat.v1.variable_scope("cost", reuse=tf.compat.v1.AUTO_REUSE):
            self.num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
            self.dim_mult = 2 if self.concat else 1

            self.current = layers.Dense((self.dim_mult * self.dims[-1] * 2), self.dim_mult * self.dims[-1],
                                        dropout=self.placeholders['dropout'],
                                        act=lambda x: x)

            self.node_pred = layers.Dense((self.dim_mult * self.dims[-1]), 1,
                                          dropout=self.placeholders['dropout'],
                                          act=lambda x: x)
            self.aggregators = self.build_aggregators(self.num_samples, self.dims, concat=self.concat,
                                                      model_size=self.model_size)

    def forward(self, h, t):
        htm1 = h[0]
        tmp_h = h[1]
        ltm1 = h[2]


        t_seq = self.seq[t]

        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos, t)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos, t)

        samples1 = self.gather_list(t_seq, samples1)
        samples2 = self.gather_list(t_seq, samples2)
        # sam1 = tf.compat.v1.Print(samples1, [samples1])

        outputs1 = self.aggregate2(samples1, self.dims, self.num_samples,
                                   support_sizes1, self.aggregators, concat=self.concat, model_size=self.model_size)
        # 'h_t_transposed/while/gcnaggregator_2/MatMul:0'
        outputs2 = self.aggregate2(samples2, self.dims, self.num_samples,
                                   support_sizes2, self.aggregators, concat=self.concat, model_size=self.model_size)
        self.outputs1 = tf.nn.l2_normalize(outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(outputs2, 1)
        h_t = self.current(tf.concat((self.outputs1, self.outputs2), -1))

        label = tf.reshape(tf.cast(tf.less_equal(self.placeholders['labels'], t), tf.float32), [-1, 1])
        loss = self._loss(label, h_t)
        return [h_t, htm1, loss]

    # @tf.function
    def _loss(self, label, h_t):
        node_preds = self.node_pred(h_t)
        # label = tf.compat.v1.Print(label, [label])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=node_preds,
            labels=tf.stop_gradient(label)))
        return loss
    