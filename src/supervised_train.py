from __future__ import division
from __future__ import print_function
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics
from Hypothesis_train import Hypo_Gen
tf.compat.v1.disable_eager_execution()

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
tf.compat.v1.app.flags.DEFINE_boolean('log_device_placement', False,
                                      """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'gcn', 'model names. See README for possible values.')
flags.DEFINE_string('rnn_model', 'sm', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate.')
flags.DEFINE_float("train_ratio", 0.2, "ratio of training set")
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string("key_type", "all_keys", "dataset category to use")
flags.DEFINE_string("test_type", "all", "test method to use")
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.2, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 20, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('neg_sample1', 20, 'minibatch size.')
flags.DEFINE_integer('neg_sample2', 40, 'minibatch size.')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 256, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', True, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 25600, "how many nodes per validation sample.")
flags.DEFINE_integer('print_every', 40, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")



flags.DEFINE_string("data_folder", "data/Virology", "dataset to use")  # node 38547
flags.DEFINE_integer('year_interval', 10, "时间间隔")
flags.DEFINE_integer('year_start', 1959, "start")
flags.DEFINE_integer('year_end', 1979, "end")  # 45w/199w
flags.DEFINE_string('save_path', "./Model/Vir/sm", "save path")
# flags.DEFINE_string('save_path', "./Model/Vir/dl", "save path")
# flags.DEFINE_string('save_path', "./Model/Vir/dg", "save path")
# flags.DEFINE_string('save_path', "./Model/Vir/dsum", "save path")

def main(argv=None):
    verbose = True
    tf.compat.v1.reset_default_graph()
    hpg = Hypo_Gen(verbose)
    hpg.train(FLAGS, verbose)

if __name__ == '__main__':
    seed = 2021
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
    tf.random.set_seed(seed)
    tf.compat.v1.app.run()
    main()