# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, time
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python import debug as tf_dbg
import networkx as nx

glo_seed = 2021
import random as rn
import pdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np
import itertools
import math
from scipy.sparse import hstack, csr_matrix, lil_matrix, vstack
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss, silhouette_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import load_npz, save_npz, csr_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer, normalize
from sklearn.model_selection import StratifiedShuffleSplit
import itertools, tqdm
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import shutil
from collections import OrderedDict
from sklearn import metrics

from models_util import SAGEInfo
from minibatch import MinibatchIterator
from neigh_samplers import UniformNeighborSampler
from multiprocessing import Pool, Value
from functools import partial
import copy
from matplotlib.backends.backend_pdf import PdfPages

tf.compat.v1.disable_eager_execution()

np.random.seed(glo_seed)
rn.seed(glo_seed)
os.environ['PYTHONHASHSEED'] = str(glo_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(glo_seed)

class Hypo_Gen():
    def __init__(self, verbose):
        """
        Initializer
        dim_Av -- dimension of node features
        dim_Ae -- dimension of edge features
        dim_y -- number of classes
        l_dim -- hidden layer dimension
        num_neighb --  No. of neighbors to consider per node (e.g. 10)
        lrate -- learning rate (e.g. 1e-3) for Adam
        model_dir -- directory to save the trained models
        """
        self.verbose = verbose
        self.globi = 0

        rn.seed(glo_seed)
        self.rng = np.random.RandomState(seed=glo_seed)
        np.random.seed(glo_seed)
        tf.compat.v1.set_random_seed(glo_seed)

        rn.seed(glo_seed)
        self.rng = np.random.RandomState(seed=glo_seed)
        np.random.seed(glo_seed)

    def get_neg_assume(self, item, full_ids, ):
        rez = []
        # print(item)
        key, value = item
        diff = list(full_ids - value - {key})
        # print(diff)
        if len(diff) > self.max_neg_per_node:
            np.random.seed(glo_seed)
            neg = np.random.choice(diff, self.max_neg_per_node, replace=False)
        else:
            neg = diff
        return key, neg

    def sample_by_degree(self, p, n):
        """
        p: probability distribution
        n: sample n nodes
        """
        p_vals = np.array(list(p.values())).cumsum()
        p_keys = list(p.keys())
        nodes = []
        node = 0
        for i in range(n):
            rn = np.random.random()
            for j, _ in enumerate(p_vals):
                if rn < _:
                    node = j
                    break
            nodes.append(p_keys[node])
        return nodes

    def sample_neighbors(self, new_node_id_map, node_year_list, cur_year):

        node_count = len(self.G.nodes())
        plus128 = 0
        len10_20 = 0
        constrain_year = 3

        node_neighbs = []
        for i in range(node_count):
            # print(i)
            neighbs = [n for n in self.G.neighbors(i)]
            d_mtx = {_: self.G.degree(_) for _ in neighbs}
            d_sum = np.sum(list(d_mtx.values()))
            _p_dist = {n: d_mtx[n] / d_sum for n in d_mtx}
            _p_dist = dict(
                sorted(
                    _p_dist.items(),
                    key=lambda k: k[1]
                )
            )

            if len(neighbs) > self.max_degree:
                plus128 += 1
                neighbs = rn.sample(neighbs, self.max_degree)
            elif len(neighbs) < self.max_degree and len(neighbs) > 0:
                len10_20 += len(neighbs) <= 20
                neighbs = rn.choices(neighbs, k=self.max_degree)
            else:
                if i in new_node_id_map:
                    neighbs = new_node_id_map[i]
                    d_mtx = {_: self.G.degree(_) for _ in neighbs}
                    d_sum = np.sum(list(d_mtx.values()))
                    _p_dist = {n: d_mtx[n] / d_sum for n in d_mtx}
                    _p_dist = dict(
                        sorted(
                            _p_dist.items(),
                            key=lambda k: k[1]
                        )
                    )
                    if len(neighbs) > self.max_degree:
                        plus128 += 1
                        neighbs = rn.sample(neighbs, self.max_degree)
                    elif len(neighbs) < self.max_degree and len(neighbs) > 0:
                        len10_20 += len(neighbs) <= 20
                        neighbs = rn.choices(neighbs, k=self.max_degree)
                    else:
                        neighbs = [i] * self.max_degree
                else:
                    neighbs = [i] * self.max_degree

            node_neighbs.append(neighbs)

        print('128+', plus128, node_count, plus128 / node_count)
        print('20-', len10_20, node_count, len10_20 / node_count)
        return np.array(node_neighbs)

    def setup_data(self, key_type, test_ratio, test_type='eval', data_folder='cancer_data', FLAGS=None):

        edge_list = []
        id_map, y_map = {}, {}
        fut_set = OrderedDict()
        data = OrderedDict()
        nodes = set()
        _id, y_id = 0, 1
        train_mask = set()
        pair_id, mask_id = 0, 0
        node_neg = []
        node_neighbs = []
        node_year_list = []
        pair_tracker = {}
        known = set()
        year_neg_adj, year_node_neighbs = [], []
        nodes = set()
        year_mask = {}
        acp_opt = set(['none', 'any', 'full', 'all', 'new', 'neg', 'eval'])
        new_nodes = []
        test_mask = {test_set: set() for test_set in acp_opt}
        neg_test_mask = {test_set: set() for test_set in acp_opt}
        app_year = {}
        prev_con = set()

        years = range(self.start_year, self.end_year + self.year_interval + 1, self.year_interval)
        window_id = 0  # len(years) -1
        self.num_year_window = len(years) - 1
        print('num_year_window=', self.num_year_window)
        for i in range(len(years)):
            for line in open("../{}/id_maps/id_map_text_{}_{}".format(data_folder, key_type, years[i])):
                pid = line.rstrip('\n')
                try:
                    if pid not in id_map:
                        id_map[pid] = _id
                        node_year_list.append([_id, years[i]])
                        nodes.add(_id)
                        app_year[_id] = i
                        _id += 1
                    full_ids.add(id_map[pid])
                except:
                    pass
        # 此时，所有年份的id_map以及全部读取
        next_cur_fut_conn = {}
        pool = Pool(15)
        for y in range(len(years[:-1])):  # 不包括最后测试的年份
            year = years[y]
            print(year)
            full_ids = set()
            new_node_dic = {}
            cur_fut_conn = next_cur_fut_conn
            next_cur_fut_conn = {}

            edge_list = []
            test_nodes = set([])

            if year == self.start_year:
                for line in open('../{}/graph/KWgraph_{}_{}.edgelist'.format(data_folder, key_type, year)):
                    content = line.rstrip('\n')
                    s_pid, t_pid, _ = content.split(" ")
                    edge_list.append([id_map[s_pid], id_map[t_pid]])
                    if id_map[s_pid] in cur_fut_conn:
                        cur_fut_conn[id_map[s_pid]].add(id_map[t_pid])
                    else:
                        cur_fut_conn[id_map[s_pid]] = {id_map[t_pid]}

                    if id_map[t_pid] in cur_fut_conn:
                        cur_fut_conn[id_map[t_pid]].add(id_map[s_pid])
                    else:
                        cur_fut_conn[id_map[t_pid]] = {id_map[s_pid]}
                self.G = nx.Graph(edge_list)
                # 画图
            else:
                self.G = self.G_future
            # pdb.set_trace()
            # nx.draw(self.G,with_labels=True,font_color='#000000',node_color = 'r',font_size =8,node_size =20)
            # plt.savefig('./log/'+str(year)+'.png')
            edge_list_future = []
            for line in open('../{}/graph/KWgraph_{}_{}.edgelist'.format(data_folder, key_type, years[y + 1])):
                content = line.rstrip('\n')
                try:
                    s_pid, t_pid, _ = content.split(" ")
                except:
                    pdb.set_trace()
                edge_list_future.append([id_map[s_pid], id_map[t_pid]])
                if id_map[s_pid] in next_cur_fut_conn:
                    next_cur_fut_conn[id_map[s_pid]].add(id_map[t_pid])
                else:
                    next_cur_fut_conn[id_map[s_pid]] = {id_map[t_pid]}  # set

                if id_map[t_pid] in next_cur_fut_conn:
                    next_cur_fut_conn[id_map[t_pid]].add(id_map[s_pid])
                else:
                    next_cur_fut_conn[id_map[t_pid]] = {id_map[s_pid]}

            self.G_future = nx.Graph(edge_list_future)
            graph_nodes = set(self.G.nodes())

            if year == self.end_year:
                self.max_neg_per_node = FLAGS.neg_sample2
            else:
                self.max_neg_per_node = FLAGS.neg_sample1

            cur_nodes = self.G.nodes()
            for node in cur_nodes:
                year_pid = "{}_{}".format(node, year)
                y_map[year_pid] = y_id
                y_id += 1
            # if y == 2:
            #     pdb.set_trace()
            node_difference = set(self.G_future.nodes()) - set(cur_nodes)
            new_node_id_map = {}
            node_icd = set()
            for node in node_difference:
                neighbors = set([n for n in self.G_future.neighbors(node)])
                avail_in_graph = graph_nodes & neighbors
                if len(avail_in_graph) > 0:
                    year_pid = "{}_{}".format(node, year)
                    new_node_dic[year_pid] = [y_map["{}_{}".format(n, year)] for n in avail_in_graph]
                    new_node_id_map[node] = list(avail_in_graph)
                    y_map[year_pid] = y_id
                    node_icd.add(node)
                    y_id += 1
            new_nodes.append(new_node_dic)

            for e in self.G_future.edges():
                s, t = e
                if not self.G.has_edge(s, t):
                    if s in cur_fut_conn:
                        cur_fut_conn[s].add(t)
                    else:
                        cur_fut_conn[s] = {t}

                    if t in cur_fut_conn:
                        cur_fut_conn[t].add(s)
                    else:
                        cur_fut_conn[t] = {s}
                    ## t+1时间步新增的边加入
                    frwd, bck = "{}_{}".format(s, t), "{}_{}".format(t, s)
                    if app_year[s] <= window_id + 1 and app_year[t] <= window_id + 1:
                        apd = window_id
                    else:
                        apd = 100
                    if frwd not in pair_tracker and bck not in pair_tracker:
                        pair_tracker[frwd] = pair_id  # pair_tracker似乎不会记录第一个时间点的边
                        cur_pair_id = pair_id

                        data[cur_pair_id] = [s, t, window_id, apd]  # apd表示两个节点的验证时间，window_id表示发现关系的时间
                        pair_id += 1
                        added = True
                    elif frwd in pair_tracker:
                        cur_pair_id = pair_tracker[frwd]
                        data[cur_pair_id][-2] = window_id  # 更新以前的关系验证的时间
                        if data[cur_pair_id][-1] > apd:
                            data[cur_pair_id][-1] = apd

                        added = False
                    else:
                        cur_pair_id = pair_tracker[bck]
                        data[cur_pair_id][-2] = window_id
                        if data[cur_pair_id][-1] > apd:
                            data[cur_pair_id][-1] = apd

                        added = False

                    if year == self.end_year:
                        if added == False:
                            if cur_pair_id in train_mask:
                                train_mask.remove(cur_pair_id)
                                # pass
                        if test_type != 'eval':
                            if (s in graph_nodes) and (t in graph_nodes):
                                test_mask['all'].add(cur_pair_id)  # Transductive
                            if (s in graph_nodes) or (t in graph_nodes):
                                test_mask['any'].add(cur_pair_id)  # Inductive 至少有一个见过
                            if not ((s in graph_nodes) or (t in graph_nodes)):
                                test_mask['none'].add(cur_pair_id)  # Inductive两个都没见过
                            if (s in node_difference) or (t in node_difference):
                                test_mask['new'].add(cur_pair_id)
                            test_mask['full'].add(cur_pair_id)
                        else:
                            new_eval_pairs = open("../{}/eval_ids.txt".format(data_folder), 'rU').read().split('\n')[
                                             :-1]
                            for node_pair in new_eval_pairs:
                                es_pid, et_pid, yr = node_pair.split(" ")
                                s, t = id_map[es_pid], id_map[et_pid]

                                frwd, bck = "{}_{}".format(s, t), "{}_{}".format(t, s)

                                if frwd not in pair_tracker and bck not in pair_tracker:
                                    pair_tracker[frwd] = pair_id
                                    if app_year[s] <= window_id + 1 and app_year[t] <= window_id + 1:
                                        apd = window_id
                                    else:
                                        apd = 100
                                    if int(yr) <= year + self.year_interval:
                                        data[pair_id] = [s, t, window_id, apd]
                                    else:
                                        data[pair_id] = [s, t, 100, apd]
                                    test_mask['eval'].add(pair_id)
                                    pair_id += 1
                                elif frwd in pair_tracker:
                                    test_mask['eval'].add(pair_tracker[frwd])
                                    if cur_pair_id in train_mask:
                                        train_mask.remove(cur_pair_id)
                                else:
                                    test_mask['eval'].add(pair_tracker[bck])
                                    if cur_pair_id in train_mask:
                                        train_mask.remove(cur_pair_id)

                            cur_fut_conn = {}
                            break

                    else:
                        if (s in graph_nodes) and (t in graph_nodes):
                            train_mask.add(cur_pair_id)
                            if year < self.end_year - self.year_interval:
                                prev_con.add(cur_pair_id)
                            mask_id += 1

            # pdb.set_trace()
            self.max_id = _id
            edge_list = []
            if year == self.end_year:
                full_ids = set(self.G_future.nodes())
            else:
                full_ids = set(self.G.nodes())
            order = []
            node_pair_list = list(cur_fut_conn.items())
            full_ids1 = full_ids
            node_pair_list1 = node_pair_list
            u_set = pool.map(partial(self.get_neg_assume, full_ids=full_ids), node_pair_list)

            # 当前节点，下一时间点的邻居
            for item in u_set:
                for x in item[1]:
                    if app_year[x] <= window_id + 1 and app_year[item[0]] <= window_id + 1:
                        apd = window_id
                    else:
                        apd = 100
                    frwd = "{}_{}".format(item[0], x)
                    bck = "{}_{}".format(x, item[0])

                    if frwd not in pair_tracker and bck not in pair_tracker:
                        pair_tracker[frwd] = pair_id
                        cur_pair_id = pair_id
                        data[cur_pair_id] = [item[0], x, 100, apd]  # apd为x验证的时间，100 表示二者没有关系
                        pair_id += 1
                        added = True
                    elif frwd in pair_tracker:
                        cur_pair_id = pair_tracker[frwd]
                        if data[cur_pair_id][-1] > apd:
                            data[cur_pair_id][-1] = apd
                        added = False

                    else:
                        cur_pair_id = pair_tracker[bck]
                        if data[cur_pair_id][-1] > apd:
                            data[cur_pair_id][-1] = apd

                        added = False

                    if added == False:
                        if cur_pair_id in train_mask:
                            train_mask.remove(cur_pair_id)
                            # pass

                    if year == self.end_year:
                        if (item[0] in graph_nodes) and (x in graph_nodes):
                            neg_test_mask['all'].add(cur_pair_id)
                        if (item[0] in graph_nodes) or (x in graph_nodes):
                            neg_test_mask['any'].add(cur_pair_id)
                        if (item[0] not in graph_nodes) or (x not in graph_nodes):
                            neg_test_mask['new'].add(cur_pair_id)

                        neg_test_mask['full'].add(cur_pair_id)
                    else:
                        train_mask.add(cur_pair_id)

                order.append(item[0])
            self.G.add_nodes_from(set(range(_id)) - set(self.G.nodes))
            year_node_neighbs.append(self.sample_neighbors(new_node_id_map, node_year_list, years[window_id]))
            window_id += 1

        pool.close()
        pool.join()

        self.id_map = id_map
        self.adj = np.array(year_node_neighbs)
        self.seq = np.transpose(self.create_series_data(nodes, y_map))

        data = np.vstack(data.values())
        data[list(prev_con), 2] = 100
        train_mask=list(train_mask)

        return data, [train_mask, test_mask, neg_test_mask, known], y_map, new_nodes

    def create_series_data(self, nodes, id_map):
        nodes = list(nodes)
        year_winds = range(self.start_year, self.end_year + 1, self.year_interval)
        seq = np.zeros((len(nodes), len(year_winds)), dtype=np.int32)
        for i in range(len(year_winds)):
            for j in range(len(nodes)):
                pid = "{}_{}".format(nodes[j], year_winds[i])
                if pid in id_map:
                    seq[j, i] = id_map[pid]

        return seq

    def get_node_context_attribute(self, id_map, feature_dim):

        context_arr = np.zeros((len(id_map) + 1, feature_dim))
        years = range(self.start_year, self.end_year + 1, self.year_interval)

        for year in years:
            node_context = load_npz(
                "../{}/emb/lsi/edge_lsi_{}_{}_emb.npz".format(self.FLAGS.data_folder, self.FLAGS.key_type,
                                                              year)).toarray()
            edge_content_file = "../{}/edge_context/edge_context_{}_{}".format(self.FLAGS.data_folder,
                                                                               self.FLAGS.key_type, year)

            # print(node_context.shape)
            for line in open(edge_content_file):
                context = line.rstrip('\n')
                pid, text = context.split(" ")
                indx = text.split(",")
                o_pid = "{}_{}".format(self.id_map[pid], year)
                if len(indx) > 0:
                    try:
                        context_arr[id_map[o_pid]] = node_context[np.array(indx).astype(np.int32) - 1, :].mean(0)
                    except:
                        pdb.set_trace()

        return context_arr

    def calc_f1(self, y_true, y_pred, is_test=False):
        y_pred_ = (y_pred >= 0.5).astype(np.float32)
        y_true = (y_true <= self.num_year_window).astype(np.float32)
        f1_mac = metrics.f1_score(y_true, y_pred_, average="macro")
        acc = metrics.accuracy_score(y_true, y_pred_)
        f1_bi = metrics.f1_score(y_true, y_pred_, average="binary")
        reverse_ytrue = ~(y_true.astype(np.bool))
        reverse_ytrue = reverse_ytrue.astype(np.float32)
        print('val_count={},max_pred={},min_pred={},avg_pred={},p_m={},n_m={}'.format(
            len(np.nonzero(y_true)[0]), max(y_pred), min(y_pred), np.mean(y_pred),
            np.mean(np.multiply(y_pred, np.expand_dims(y_true, axis=1))),
            np.mean(np.multiply(y_pred, np.expand_dims(reverse_ytrue, axis=1)))
        ))
        if is_test == False:
            return f1_mac, acc, f1_bi, None
        else:
            f1_bi = metrics.f1_score(y_true, y_pred_, average="binary")
            auc = metrics.roc_auc_score(y_true, y_pred)
            return f1_mac, acc, f1_bi, auc

    def conf_mat(self, y_true, y_pred):
        y_pred = (y_pred >= 0.5).astype(np.float32)
        y_true = (y_true <= self.num_year_window).astype(np.float32)
        return confusion_matrix(y_true, y_pred)

    # Define model evaluation function
    def evaluate(self, sess, model, minibatch_iter, size=None):
        t_test = time.time()
        minibatch_iter.set_eval_data(size)
        iter_num = 0
        finished = False
        labels=[]
        preds=[]
        costs=[]
        while not finished:
            feed_dict_val, finished = minibatch_iter.generate_batch_eval_data(iter_num, self.FLAGS.batch_size)
            node_outs_val = sess.run([model.preds, model.loss],
                                     feed_dict=feed_dict_val[0])
            labels.append(feed_dict_val[1])
            preds.append(node_outs_val[0])
            costs.append(node_outs_val[1])
            iter_num+=1
        preds=np.vstack(preds)
        labels=np.hstack(labels)
        f1max, acc, f1bin, _ = self.calc_f1(labels, preds)

        return np.mean(costs), f1max, acc, f1bin, (time.time() - t_test)

    def safe_div(self, n, d, ret_val=0):
        return n / d if d else ret_val

    def log_dir(self):
        log_dir = self.FLAGS.base_log_dir + "/sup-" + self.FLAGS.train_prefix.split("/")[-2]
        log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=self.FLAGS.model,
            model_size=self.FLAGS.model_size,
            lr=self.FLAGS.learning_rate)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    # def get_pair_indx(self, sess, model, minibatch_iter, size):
    #     finished = False
    #     y_pred = []
    #     y_true = []
    #     X = []
    #     iter_num = 0
    #     finished = False
    #     while not finished:
    #         feed_dict_val, batch_labels, finished, val_edges, _ = minibatch_iter.incremental_val_feed_dict(size,
    #                                                                                                        iter_num,
    #                                                                                                        test=True)
    #         node_outs_val = sess.run([model.preds],
    #                                  feed_dict=feed_dict_val)
    #         y_pred.append(node_outs_val[0])
    #         y_true.append(batch_labels)
    #         X.append(val_edges)
    #         iter_num += 1
    #     y_pred = np.vstack(y_pred)
    #     y_true = np.vstack(y_true)
    #     X = np.vstack(X)
    #     y_pred[y_pred > 0.5] = 1
    #     y_pred[y_pred <= 0.5] = 0

    #     comb = y_pred * (y_true + y_pred)

    #     correct_pred = np.random.choice(np.where(comb == 2)[0], 20, replace=False)
    #     cand = np.random.choice(np.where(comb == 1)[0], 20, replace=False)

    #     return X[cand], X[correct_pred]

    def incremental_evaluate(self, sess, model, minibatch_iter, size, test=False, test_set='full'):
        t_test = time.time()
        finished = False
        losses = []
        val_preds = []
        labels = []
        iter_num = 0
        finished = False
        # minibatch_iter.set_test_data(test_set, test)
        while not finished:
            feed_dict_val, batch_labels, finished, _, _, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num,
                                                                                                      test=test)
            #
            # feed_dict_val.update({['dropout']: self.FLAGS.dropout})
            # feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})
            node_outs_val = sess.run([model.preds, model.loss, model.pre_preds],
                                     feed_dict=feed_dict_val)
            val_preds.append(node_outs_val[0])
            labels.append(batch_labels)
            losses.append(node_outs_val[1])
            iter_num += 1
        val_preds = np.vstack(val_preds)
        labels = np.hstack(labels)
        f1_scores = self.calc_f1(labels, val_preds, test)
        if test == True:
            l = (labels <= self.num_year_window).astype(np.float32)
            val_preds_ = (val_preds >= 0.5).astype(np.float32)
            pres, rec = precision_score(l, val_preds_, ), recall_score(l, val_preds_, )
            print('label_shape={},l_nonzero={},pred_nonzero={},max_preds={},min_preds={}'.format(
                l.shape, len(np.nonzero(l)[0]), len(np.nonzero(val_preds_)[0]), max(val_preds), min(val_preds)
            ))
            print("press {}, rec {}".format(pres, rec))

        return np.mean(losses), f1_scores[0], f1_scores[1], f1_scores[2], f1_scores[3], (time.time() - t_test)

    def construct_placeholders(self, num_year_window):
        # Define placeholders
        placeholders = {
            'labels': tf.compat.v1.placeholder(tf.int32, shape=(None,), name='labels'),
            'batch1': tf.compat.v1.placeholder(tf.int32, shape=(None,), name='batch1'),
            'batch2': tf.compat.v1.placeholder(tf.int32, shape=(None,), name='batch2'),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
            'batch_size': tf.compat.v1.placeholder(tf.int32, name='batch_size'),
            'is_train': tf.compat.v1.placeholder(tf.bool, name='is_train'),
            'length': tf.compat.v1.placeholder(tf.int32, name='length'),
            'lr': tf.compat.v1.placeholder(tf.float32, name='lr')
        }
        return placeholders

    def pos2vec(self, years, model_size):
        pos = [_ for _ in range(len(years))]
        pos = np.array(pos)
        print(pos)
        PE = np.zeros((len(pos), model_size))
        for i in pos:
            for j in range(model_size):
                if j % 2 == 0:
                    PE[i, j] = np.sin(pos[i] / 10000 ** (j / model_size))
                else:
                    PE[i, j] = np.cos(pos[i] / 10000 ** ((j - 1) / model_size))
        PE = tf.constant(PE, dtype=tf.float32)
        return PE

    def logging(self, s, print_=True, log_=True):  #
        if print_:
            print(s)
        if log_:
            with open(os.path.join(os.path.join("logs", self.FLAGS.save_path.split('/')[-2],
                                                        self.FLAGS.save_path.split('/')[-1])), 'a+') as f_log:
                f_log.write(s + '\n')

    def train(self, FLAGS, verbose=False):

        num_classes = 1
        self.year_interval = FLAGS.year_interval
        self.max_degree = FLAGS.max_degree
        self.start_year = FLAGS.year_start
        self.end_year = FLAGS.year_end
        feature_dim = 300
        self.cid = ('{0}_{1}_{2}_{3}_{4}_{5}'.format(
            FLAGS.data_folder,
            FLAGS.model,
            FLAGS.epochs,
            FLAGS.learning_rate,
            self.end_year,
            time.time()))
        data, masks, id_map, new_nodes = self.setup_data(FLAGS.key_type, FLAGS.train_ratio, FLAGS.test_type,
                                                         FLAGS.data_folder, FLAGS)
        # id_map, 其实是返回的y_map
        self.FLAGS = FLAGS
        years = range(self.start_year, self.end_year + 1, self.year_interval)

        p2v = self.pos2vec(years, 256)
        features = np.zeros((len(id_map) + 1, feature_dim))
        for year in years:
            tmp_features = load_npz(
                "../{}/emb/lsi/node_lsi_{}_{}_emb.npz".format(self.FLAGS.data_folder, self.FLAGS.key_type,
                                                              year)).toarray()
            ids = map(lambda x: id_map["{}_{}".format(self.id_map[x], year)], open(
                "../{}/id_maps/id_map_text_{}_{}".format(self.FLAGS.data_folder, self.FLAGS.key_type,
                                                         year)).read().split("\n")[:-1])
            features[list(ids), :] = tmp_features

        context_features = self.get_node_context_attribute(id_map, feature_dim)
        if self.FLAGS.random_context:
            features = np.add(features, context_features)
        for i in range(len(new_nodes)):
            for node, neighbors in new_nodes[i].items():
                features[id_map[node]] = features[neighbors[:20]].mean()

        labels = data[:, 2]
        print(labels.shape, max(labels), min(labels))

        app_label = data[:, 3]
        data = data[:, :2]

        placeholders = self.construct_placeholders(len(years) + 1)
        minibatch = MinibatchIterator(data, labels, app_label, masks, self.adj,
                                      placeholders,
                                      self.seq.shape[0],
                                      fut_set=set(list([])),
                                      max_id=self.max_id,
                                      batch_size=self.FLAGS.batch_size,
                                      max_degree=self.FLAGS.max_degree)

        adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=self.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        seq_info_ph = tf.compat.v1.placeholder(tf.int32, shape=self.seq.shape)
        seq_info = tf.Variable(seq_info_ph, trainable=False, name="seq_info")

        if self.FLAGS.rnn_model == 'sm':
            from supervised_models import SupervisedGraphsage
        elif self.FLAGS.rnn_model == 'dl':
            from rnn_ht_diff_local import SupervisedGraphsage
        elif self.FLAGS.rnn_model == 'dg':
            from rnn_ht_diff_global import SupervisedGraphsage
        elif self.FLAGS.rnn_model == 'dsum':
            from rnn_ht_diff_sum import SupervisedGraphsage
        elif self.FLAGS.rnn_model == 'lstm':
            from lstm import SupervisedGraphsage

        if self.FLAGS.model == 'graphsage_mean':
            # Create model
            sampler = UniformNeighborSampler(adj_info)
            if self.FLAGS.samples_3 != 0:
                layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                               SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2),
                               SAGEInfo("node", sampler, self.FLAGS.samples_3, self.FLAGS.dim_2)]
            elif self.FLAGS.samples_2 != 0:
                layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                               SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]
            else:
                layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1)]

            model = SupervisedGraphsage(data, labels, self.seq.shape[0], placeholders,
                                        features,
                                        context_features,
                                        seq_info,
                                        layer_infos=layer_infos,
                                        model_size=self.FLAGS.model_size,
                                        sigmoid_loss=self.FLAGS.sigmoid,
                                        identity_dim=self.FLAGS.identity_dim,
                                        logging=True)
        elif self.FLAGS.model == 'gcn':
            # Create model
            sampler = UniformNeighborSampler(adj_info)  ## 不让看
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, 2 * self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, 2 * self.FLAGS.dim_2)]

            model = SupervisedGraphsage(data, labels, self.seq.shape[0], placeholders, p2v,
                                        features,
                                        context_features,
                                        seq_info,
                                        layer_infos=layer_infos,
                                        aggregator_type="gcn",
                                        model_size=self.FLAGS.model_size,
                                        concat=False,
                                        sigmoid_loss=self.FLAGS.sigmoid,
                                        identity_dim=self.FLAGS.identity_dim,
                                        logging=True)

        elif self.FLAGS.model == 'graphsage_seq':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SupervisedGraphsage(data, labels, self.seq.shape[0], placeholders,
                                        features,
                                        context_features,
                                        seq_info,
                                        layer_infos=layer_infos,
                                        aggregator_type="seq",
                                        model_size=self.FLAGS.model_size,
                                        sigmoid_loss=self.FLAGS.sigmoid,
                                        identity_dim=self.FLAGS.identity_dim,
                                        logging=True)

        elif self.FLAGS.model == 'graphsage_maxpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SupervisedGraphsage(data, labels, self.seq.shape[0], placeholders, p2v,
                                        features,
                                        context_features,
                                        seq_info,
                                        layer_infos=layer_infos,
                                        aggregator_type="maxpool",
                                        model_size=self.FLAGS.model_size,
                                        sigmoid_loss=self.FLAGS.sigmoid,
                                        identity_dim=self.FLAGS.identity_dim,
                                        logging=True)

        elif self.FLAGS.model == 'graphsage_meanpool':
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [SAGEInfo("node", sampler, self.FLAGS.samples_1, self.FLAGS.dim_1),
                           SAGEInfo("node", sampler, self.FLAGS.samples_2, self.FLAGS.dim_2)]

            model = SupervisedGraphsage(data, labels, self.seq.shape[0], placeholders,
                                        features,
                                        context_features,
                                        seq_info,
                                        layer_infos=layer_infos,
                                        aggregator_type="meanpool",
                                        model_size=self.FLAGS.model_size,
                                        sigmoid_loss=self.FLAGS.sigmoid,
                                        identity_dim=self.FLAGS.identity_dim,
                                        logging=True)

        else:
            raise Exception('Error: model name unrecognized.')

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True

        sess = tf.compat.v1.Session(config=config)

        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer(), feed_dict={adj_info_ph: self.adj,
                                                                         seq_info_ph: self.seq})
        sess.run(model.features.assign(model.features_plhdr), feed_dict={model.features_plhdr: features})
        tenboard_dir = './tensorboard/'
        # 指定一个文件用来保存图
        writer = tf.compat.v1.summary.FileWriter(tenboard_dir + str(FLAGS.learning_rate) + 'data',  # + 'ht_local',
                                                 sess.graph)
        # 把图add进去
        # writer.add_graph(sess.graph)
        total_steps = 0
        avg_time = 0.0
        cost = 0.0
        val_f1_init = 0.0
        best_epoch = 0

        train_adj_info = tf.compat.v1.assign(adj_info, self.adj)
        sess.graph.finalize()
        self.lr = self.FLAGS.learning_rate
        # 在此处设置验证集
        minibatch.set_test_data()
        self.logging('-' * 79)
        for epoch in range(self.FLAGS.epochs):
            iter = 0
            t_costs = []
            minibatch.shuffle()

            t = time.time()
            pbar = tqdm.tqdm(total=len(minibatch.train_links))
            while not minibatch.end():
                pbar.update(self.FLAGS.batch_size)
                feed_dict, labels = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: self.FLAGS.dropout})
                feed_dict.update({placeholders['lr']: self.lr})
                # Training step
                if verbose and minibatch.end():
                    outs = sess.run([model.opt_op, model.loss, model.merged], feed_dict=feed_dict)
                    train_cost = outs[1]
                    t_costs.append(train_cost)
                else:
                    outs = sess.run([model.opt_op, model.loss, model.merged], feed_dict=feed_dict)
                    t_costs.append(outs[1])
                writer.add_summary(outs[2], total_steps)

                iter += 1
                total_steps += 1

                if total_steps > self.FLAGS.max_total_steps:
                    break

            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            pbar.close()
            # print(t_costs)

            if verbose:
                if self.FLAGS.validate_batch_size == -1:
                    cost, val_f1_mac, val_acc, val_f1_bin, _, duration = self.incremental_evaluate(sess, model,
                                                                                                   minibatch,
                                                                                                   self.FLAGS.batch_size)
                else:
                    cost, val_f1_mac, val_acc, val_f1_bin, duration = self.evaluate(sess, model, minibatch,
                                                                                    self.FLAGS.validate_batch_size)
                # pdb.set_trace()
                self.logging("epoch {:04d} | Iter {:04d} | train_loss {:.5f} | val_loss {:.5f} | val_f1_mac {:.5f} | "
                             "val_f1_bin {:.5f} | val_acc {:.5f} | time {:.5f} ".format(
                        epoch, iter, sum(t_costs) / len(t_costs), cost, val_f1_mac, val_f1_bin, val_acc, avg_time
                    )
                )
            if val_f1_bin >= val_f1_init:
                saver.save(sess, self.FLAGS.save_path + "/model.ckpt")
                val_f1_init = val_f1_bin
                best_epoch = epoch

            if total_steps > self.FLAGS.max_total_steps:
                break
            # self.lr /= 1.5

        print("Optimization Finished!")
        ###
        model_file = tf.compat.v1.train.latest_checkpoint(self.FLAGS.save_path)
        saver.restore(sess, model_file)
        print("Writing test set stats to file (don't peak!)")
        for tests in ['full']:
            cost, test_f1_mac, test_acc, test_f1_bi, test_auc, duration = self.incremental_evaluate(sess, model,
                                                                                                    minibatch,
                                                                                                    self.FLAGS.batch_size,
                                                                                                    test=True,
                                                                                                    test_set=tests)
            self.logging("\n=============Result for {}============\n".format(tests.upper()))
            self.logging("best_epoch {:04d} | loss {:.5f} | f1_bi {:.5f} | f1_mac {:.5f} | "
                         "auc {:.5f} | acc {:.5f} | time {:.5f} "
                        .format(
                                best_epoch, cost, test_f1_bi, test_f1_mac, test_auc, test_acc, duration
                        )
            )

        writer.close()

