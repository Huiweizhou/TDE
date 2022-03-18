from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import models_util as models
import layers as layers
import tensorflow.keras as tfk

import numpy as np
from aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

tf.compat.v1.disable_eager_execution()

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

seed = 2021
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)


class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, pairs, labels, num_window,
                 placeholders, p2v, features, context_features, seq,  # degrees,
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
        self.length = placeholders["length"]
        self.llr = placeholders["lr"]
        self.p2v = p2v
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
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.llr)

        self.lambda_ = 0.0000001

        self.build()
        dummy_emb = tf.zeros_like(tf.expand_dims(self.inputs1, -1), tf.float32)
        h_0 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(1, self.dim_mult * self.dims[-1])),
                        name='h_0')
        tmp_h_1 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(1, self.dim_mult * self.dims[-1])),
                            name='tmp_h_1')
        # 定义第一个时刻之前的pair的label，全0
        pre_label = tf.zeros_like(tf.expand_dims(self.inputs1, -1), tf.float32)
        losses = tf.zeros_like(0.0)
        self.tmp_ht, self.g_t, _, self.losses = tf.scan(self.forward, self.range,
                                                        initializer=[h_0, tmp_h_1, pre_label,
                                                                     losses],
                                                        parallel_iterations=1,
                                                        name='h_t_transposed', swap_memory=True)

        self.loss = tf.reduce_mean(self.losses)
        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.histogram('h_t', self.g_t[-1, :, :])
        tf.compat.v1.summary.histogram('W', self.conv2d.vars['weights0'])
        tf.compat.v1.summary.histogram('bias', self.conv2d.vars['bias0'])
        self.merged = tf.compat.v1.summary.merge_all()
        grads_and_vars = self.optimizer.compute_gradients(loss=self.loss)

        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        self.pre_preds=self.g_t
        self.preds = tf.nn.sigmoid(self.node_pred(self.g_t[-1, :, :]))

        flattened_emb = tf.reshape(self.pre_preds, [-1, self.dim_mult * self.dims[-1]])
        self.all_preds = tf.reshape(tf.nn.sigmoid(self.node_pred(flattened_emb)), [num_window, -1, 1])

    def build(self):
        with tf.compat.v1.variable_scope("cost", reuse=tf.compat.v1.AUTO_REUSE):
            self.num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
            self.dim_mult = 2 if self.concat else 1

            self.recurrent = layers.Dense((self.dim_mult * self.dims[-1]), self.dim_mult * self.dims[-1],
                                          dropout=self.placeholders['dropout'],
                                          act=lambda x: x)
            self.global_recurrent = layers.Dense((self.dim_mult * self.dims[-1]), self.dim_mult * self.dims[-1],
                                                 dropout=self.placeholders['dropout'],
                                                 act=lambda x: x)

            self.global_current = layers.Dense((self.dim_mult * self.dims[-1]), self.dim_mult * self.dims[-1],
                                               dropout=self.placeholders['dropout'],
                                               act=lambda x: x)
            self.current = layers.Dense((self.dim_mult * self.dims[-1] * 3), self.dim_mult * self.dims[-1],
                                        dropout=self.placeholders['dropout'],
                                        act=lambda x: x)
            self.parma_p = layers.Dense((self.dim_mult * self.dims[-1]), self.dim_mult * self.dims[-1],
                                        dropout=self.placeholders['dropout'],
                                        act=lambda x: x)
            self.node_pred = layers.Dense((self.dim_mult * self.dims[-1]), 1,
                                          dropout=self.placeholders['dropout'],
                                          act=lambda x: x)
            self.conv2d = layers.Conv2d(dropout=self.placeholders['dropout'])

            self.aggregators = self.build_aggregators(self.num_samples, self.dims, concat=self.concat,
                                                      model_size=self.model_size)

            self.recurrent_bais = tf.Variable(name='b', initial_value=lambda: tf.initializers.ones()(
                self.dim_mult * self.dims[-1]))

            self.local_recurrent_bais = tf.Variable(name='b1', initial_value=lambda: tf.initializers.ones()(
                self.dim_mult * self.dims[-1]))
            self.global_recurrent_bais = tf.Variable(name='b2', initial_value=lambda: tf.initializers.ones()(
                self.dim_mult * self.dims[-1]))
            self.MMLP_bais = tf.Variable(name='b3', initial_value=lambda: tf.initializers.ones()(
                self.dim_mult * self.dims[-1]))
            self.t2v = layers.T2V(self.dim_mult * self.dims[-1])

    def forward(self, h, t):
        htm1 = h[0]
        tmp_h_1 = h[1]
        pre_label = h[2]
        t_seq = self.seq[t]

        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos, t)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos, t)

        samples1 = self.gather_list(t_seq, samples1)
        samples2 = self.gather_list(t_seq, samples2)

        outputs1 = self.aggregate2(samples1, self.dims, self.num_samples,
                                   support_sizes1, self.aggregators, concat=self.concat, model_size=self.model_size)
        # 'h_t_transposed/while/gcnaggregator_2/MatMul:0'
        outputs2 = self.aggregate2(samples2, self.dims, self.num_samples,
                                   support_sizes2, self.aggregators, concat=self.concat, model_size=self.model_size)
        self.outputs1 = tf.nn.l2_normalize(outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(outputs2, 1)
        label = tf.reshape(tf.cast(tf.less_equal(self.placeholders['labels'], t), tf.float32), [-1, 1])
        # t = tf.compat.v1.Print(t, [t])
        now = t
        self.time_vec = tf.tile(
            self.t2v(tf.reshape(tf.cast(now, dtype=tf.float32), [1, 1])),
            [self.length, 1]
        )
        # self.posvec = tf.tile(
        #     tf.expand_dims(self.p2v[now],0),
        #     [self.length,1]
        # )
        h_t = tf.nn.bias_add(
            self.recurrent(htm1) +
            self.current(
                tf.concat(
                    (
                        self.outputs1,
                        self.outputs2,
                        self.time_vec,
                    ), -1
                )
            ),
            self.recurrent_bais
        )
        res_tmp_h = h_t
        center_vec = tf.reduce_mean(tf.multiply(pre_label, htm1))
        center_vec = tf.reshape(center_vec, [1, -1])
        center_vec = tf.tile(
            center_vec,
            [self.length, self.dim_mult * self.dims[-1]]
        )
        global_diff = res_tmp_h - center_vec
        global_ht = tf.nn.bias_add(self.global_recurrent(global_diff
                                                         ) + self.global_current(tmp_h_1), self.global_recurrent_bais)
        h_t = global_ht
        loss = self._loss(label, h_t)

        return [res_tmp_h, h_t, label, loss]

    # @tf.function
    def _loss(self, label, h_t):
        node_preds = self.node_pred(h_t)  # 拼接输出
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=node_preds,
            labels=tf.stop_gradient(label)))

        res = tf.matmul(h_t, h_t, transpose_b=True)
        return loss + 1e-5 * (
            tf.norm(tf.multiply(tf.math.divide(1, tf.reduce_max(res)), res) - tf.eye(self.length), ord=2))
