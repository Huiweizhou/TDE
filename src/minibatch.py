from __future__ import division
from __future__ import print_function

from collections import Counter
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

seed=2021
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
class MinibatchIterator(object):

    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    train_edges -- link pairs (edges)
    """
    def __init__(self, edges, labels, app_label, masks, adj,
            placeholders, num_classes, fut_set, max_id, batch_size=100, max_degree=128,
            **kwargs):

        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.num_classes = num_classes
        self.max_id = max_id
        self.fut_set = fut_set
        self.adj = adj
        self.app_label = app_label
        self.val_size=FLAGS.validate_batch_size


        train_mask, self.test_mask, self.neg_test_mask, self.known = masks
        self.edges = edges
        self.labels = labels
        self.lastlabel = np.sort(np.unique(labels))[-2]
        self.full_train_labels = self.labels[train_mask]
        print('self.full_train_labels=', len(self.full_train_labels))
        # self.val_labels = self.full_train_labels[:FLAGS.validate_batch_size]
        self.full_train_links = self.edges[train_mask, :]
        # print('self.full_train_links=', len(self.full_train_links))

        # self.val_links = self.full_train_links[:FLAGS.validate_batch_size]
        # self.val_set_size = len(self.val_links)

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_links)

    def batch_feed_dict(self, batch_links, batch_labels, is_train=False):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_links:
            batch1.append(node1)
            batch2.append(node2)

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_links)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})
        feed_dict.update({self.placeholders['labels']: batch_labels})
        feed_dict.update({self.placeholders['is_train']: is_train})
        feed_dict.update({self.placeholders['length']: len(batch1)})

        return feed_dict, batch_labels

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_links))
        batch_links = self.train_links[start_idx : end_idx]
        batch_labels = self.train_labels[start_idx : end_idx]

        return self.batch_feed_dict(batch_links, batch_labels, True)

    def num_training_batches(self):
        return len(self.train_links) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        if size is None:
            return self.batch_feed_dict(self.val_links, self.val_labels, False)
        else:
            val_edges = self.edge_list[:size]
            val_labels = self.edge_lables[:size]
            return self.batch_feed_dict(val_edges, val_labels)

    def set_test_data(self, test_set='full', known = None, is_train=True):
        self.edge_list = np.concatenate((self.edges[list(self.test_mask[test_set])], self.edges[list(self.neg_test_mask[test_set])]), 0)
        self.edge_lables = np.concatenate((self.labels[list(self.test_mask[test_set])], self.labels[list(self.neg_test_mask[test_set])]), 0)
        self.test_app_label = np.concatenate((self.app_label[list(self.test_mask[test_set])], self.app_label[list(self.neg_test_mask[test_set])]), 0)
        if is_train:
            N = len(self.edge_list)
            ids = np.arange(N)
            ids = np.random.permutation(ids)
            self.edge_list = self.edge_list[ids]
            self.edge_lables = self.edge_lables[ids]
            self.test_app_label = self.test_app_label[ids]

    def set_eval_data(self, size):
        self.val_edges = self.edge_list[:size]
        self.val_labels = self.edge_lables[:size]

    def generate_batch_eval_data(self, iterm, batch_size):
        edges = self.val_edges[iterm*batch_size:min((iterm+1)*batch_size, self.val_size)]
        labels = self.val_labels[iterm*batch_size:min((iterm+1)*batch_size, self.val_size)]
        return self.batch_feed_dict(edges, labels), (iterm+1)*batch_size >= self.val_size

    def incremental_val_feed_dict(self, size, iter_num, test=False):
        if not test:
            self.edge_list= self.val_links
            self.edge_lables = self.val_labels
        edges = self.edge_list[iter_num*size:min((iter_num+1)*size,
            len(self.edge_list))]

        labels = self.edge_lables[iter_num*size:min((iter_num+1)*size,
            len(self.edge_lables))]

        app_labels = self.test_app_label[iter_num*size:min((iter_num+1)*size,
            len(self.test_app_label))]

        ret_val = self.batch_feed_dict(edges, labels)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(self.edge_lables), edges, labels, app_labels


    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        N = self.full_train_links.shape[0]
        ids = np.arange(N)
        ids = np.random.permutation(ids)
        self.train_links = self.full_train_links[ids]
        self.train_labels = self.full_train_labels[ids]
        print('positive label in train: ', len(self.train_labels)-Counter(self.train_labels)[100])
        self.batch_num = 0

