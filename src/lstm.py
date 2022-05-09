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
        # tmp_h_0 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(1, self.dim_mult * self.dims[-1])),
        #                 name='tmp_h_0')
        Ct = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(1, self.dim_mult * self.dims[-1])),
                        name='Ct')
        losses = tf.zeros_like(0.0)
        self.h_t, _, self.losses = tf.scan(self.forward, self.range, initializer=[h_0, Ct, losses], parallel_iterations=1,
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

            self.recurrent = layers.Dense((self.dim_mult * self.dims[-1]), self.dim_mult * self.dims[-1],
                                          dropout=self.placeholders['dropout'],
                                          act=lambda x: x)
            self.res_recurrent = layers.Dense((self.dim_mult * self.dims[-1]), self.dim_mult * self.dims[-1],
                                          dropout=self.placeholders['dropout'],
                                          act=lambda x: x)
            self.current = layers.Dense((self.dim_mult * self.dims[-1] * 2), self.dim_mult * self.dims[-1],
                                        dropout=self.placeholders['dropout'],
                                        act=lambda x: x)

            self.node_pred = layers.Dense((self.dim_mult * self.dims[-1]), 1,
                                          dropout=self.placeholders['dropout'],
                                          act=lambda x: x)
            self.aggregators = self.build_aggregators(self.num_samples, self.dims, concat=self.concat,
                                                      model_size=self.model_size)
            self.Wf = layers.Dense((self.dim_mult * self.dims[-1] * 3), self.dim_mult * self.dims[-1],
                                        dropout=self.placeholders['dropout'],
                                        act=lambda x: x)
            self.Wi = layers.Dense((self.dim_mult * self.dims[-1] * 3), self.dim_mult * self.dims[-1],
                                        dropout=self.placeholders['dropout'],
                                        act=lambda x: x)
            self.Wc = layers.Dense((self.dim_mult * self.dims[-1] * 3), self.dim_mult * self.dims[-1],
                                        dropout=self.placeholders['dropout'],
                                        act=lambda x: x)
            self.Wo = layers.Dense((self.dim_mult * self.dims[-1] * 3), self.dim_mult * self.dims[-1],
                                        dropout=self.placeholders['dropout'],
                                        act=lambda x: x)
            # self.bf = tf.Variable(name='bf', initial_value=lambda: tf.initializers.ones()(
            #     self.dim_mult * self.dims[-1]))
            # self.bi = tf.Variable(name='bi', initial_value=lambda: tf.initializers.ones()(
            #     self.dim_mult * self.dims[-1]))
            # self.bc = tf.Variable(name='bc', initial_value=lambda: tf.initializers.ones()(
            #     self.dim_mult * self.dims[-1]))
            # self.bo = tf.Variable(name='bo', initial_value=lambda: tf.initializers.ones()(
            #     self.dim_mult * self.dims[-1]))


            self.recurrent_bais = tf.Variable(name='b', initial_value=lambda: tf.initializers.ones()(
                self.dim_mult * self.dims[-1]))

            self.res_recurrent_bais = tf.Variable(name='b1', initial_value=lambda: tf.initializers.ones()(
                self.dim_mult * self.dims[-1]))
            # self.test = tf.Variable(tf.random.uniform([200, 200], minval=-1, maxval=1, dtype=tf.float32))

    def forward(self, h, t):
        htm1 = h[0]
        # tmp_h = h[1]
        Ct_1 = h[1]
        # ltm1 = h[3]


        t_seq = self.seq[t]

        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos, t)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos, t)

        samples1 = self.gather_list(t_seq, samples1)
        samples2 = self.gather_list(t_seq, samples2)

        outputs1 = self.aggregate2(samples1, self.dims, self.num_samples,
                                   support_sizes1, self.aggregators, concat=self.concat, model_size=self.model_size)

        outputs2 = self.aggregate2(samples2, self.dims, self.num_samples,
                                   support_sizes2, self.aggregators, concat=self.concat, model_size=self.model_size)
        self.outputs1 = tf.nn.l2_normalize(outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(outputs2, 1)
   
        # h_t = tf.nn.bias_add(self.recurrent(htm1) + self.current(tf.concat((self.outputs1, self.outputs2), -1)),
        #                      self.recurrent_bais)
        ft = tf.sigmoid(
                self.Wf(tf.concat((htm1, self.outputs1, self.outputs2), -1))
            )
        
        it = tf.sigmoid(
                self.Wi(tf.concat((htm1, self.outputs1, self.outputs2), -1))
            )
        C_hat = tf.tanh(
                self.Wc(tf.concat((htm1, self.outputs1, self.outputs2), -1))
            )
        C_t = tf.multiply(ft, Ct_1) + tf.multiply(it, C_hat)
        ot = tf.sigmoid(
                self.Wo(tf.concat((htm1, self.outputs1, self.outputs2), -1))
            )
        h_t=tf.multiply(ot,tf.tanh(C_t))


        label = tf.reshape(tf.cast(tf.less_equal(self.placeholders['labels'], t), tf.float32), [-1, 1])
        loss = self._loss(label, h_t)
        # return [h_t, htm1, C_t, loss]
        return [h_t, C_t, loss]

    # @tf.function
    def _loss(self, label, h_t):
        node_preds = self.node_pred(h_t)
        # label = tf.compat.v1.Print(label, [label])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=node_preds,
            labels=tf.stop_gradient(label)))
        return loss
    
    @tf.function
    def focal_loss(self, label, h_t, alpha=0.2, gamma=5):
        node_preds = tf.sigmoid(self.node_pred(h_t))
        zeros = tf.zeros_like(node_preds, dtype=node_preds.dtype)
        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
        pos_p_sub = tf.where(tf.stop_gradient(label) > zeros, label - node_preds, zeros) # positive sample 寻找正样本，并进行填充

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(tf.stop_gradient(label) > zeros, zeros, node_preds) # negative sample 寻找负样本，并进行填充
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.compat.v1.log(tf.clip_by_value(node_preds, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.compat.v1.log(tf.clip_by_value(1.0 - node_preds, 1e-8, 1.0))
        return tf.reduce_mean(per_entry_cross_ent)

