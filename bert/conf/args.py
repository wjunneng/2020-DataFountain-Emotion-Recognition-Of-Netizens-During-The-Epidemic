# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import torch
from bert.core.lstm import LSTM
from bert.core.ian import IAN
from bert.core.memnet import MemNet
from bert.core.ram import RAM
from bert.core.td_lstm import TD_LSTM
from bert.core.cabasc import Cabasc
from bert.core.atae_lstm import ATAE_LSTM
from bert.core.tnet_lf import TNet_LF
from bert.core.aoa import AOA
from bert.core.mgan import MGAN
from bert.core.lcf_bert import LCF_BERT
from bert.core.bert_spc import BERT_SPC
from bert.core.aen import AEN_BERT

# -* orignal dir/path *-
# project_dir = os.path.dirname(os.path.abspath('..'))
project_dir = '/home/wjunneng/Ubuntu/2020-DataFountain-Emotion-Recognition-Of-Netizens-During-The-Epidemic'
data_dir = os.path.join(project_dir, 'data')
input_dir = os.path.join(data_dir, 'input')
test_dataset_dir = os.path.join(input_dir, 'test_dataset')
train_dataset_dir = os.path.join(input_dir, 'train_dataset')
pretain_model_dir = os.path.join(data_dir, 'pretain_model')

# -* test *-
test_10k_path = os.path.join(test_dataset_dir, 'nCov_10k_test_utf_8.csv')
train_100k_path = os.path.join(train_dataset_dir, 'test.csv')
train_900k_path = os.path.join(train_dataset_dir, 'test.csv')

# test_10k_path = os.path.join(test_dataset_dir, 'nCov_10k_test_utf_8.csv')
# train_100k_path = os.path.join(train_dataset_dir, 'nCoV_100k_train_utf_8.labled.csv')
# train_900k_path = os.path.join(train_dataset_dir, 'nCoV_900k_train_utf_8.unlabled.csv')
submit_example_path = os.path.join(input_dir, 'submit_example.csv')

# -* generate dir/path *-
log_dir = os.path.join(data_dir, 'log')
output_dir = os.path.join(data_dir, 'output')
submit_path = os.path.join(data_dir, 'submit.csv')
labels = ['-1', '0', '1']  # 消极 中性 积极

input_categories = '微博中文内容'
target_categories = '微博发布时间'
output_categories = '情感倾向'

BATCH = 6
EPOCHS = 1
do_fold = True
# ############################### model parameters
topics = None
# 是否按照批量预测
predict_batch = True

# 模型名称
model_name = 'aen_bert'
# 优化算法
optimizer = 'adam'
# 初始化方式
initializer = 'xavier_uniform_'
# 'try 5e-5, 2e-5 for BERT, 1e-3 for others'
learning_rate = 2e-5
# 随机失活率
dropout = 0.1
# 权重
l2reg = 0.01
# 步长
log_step = 200
# 嵌入的维度
embed_dim = 300
# 隐藏层神经元个数
hidden_dim = 300
# bert 维度
bert_dim = 768
# 序例最大的长度
max_seq_len = 128
# 极性的维度
polarities_dim = 3
# hops
hops = 3
# e.g. cuda:0
device = None
# set seed for reproducibility
seed = 42
# set ratio between 0 and 1 for validation support
valset_ratio = 0.2
# local context focus mode, cdw or cdm
local_context_focus = 'cdm'
# semantic-relative-distance, see the paper of LCF-BERT model
SRD = 3

# ############################### other parameters
# default hyper-parameters for LCF-BERT model is as follws:
# lr: 2e-5
# l2: 1e-5
# batch size: 16
# num epochs: 5
model_classes = {
    'lstm': LSTM,
    'td_lstm': TD_LSTM,
    'atae_lstm': ATAE_LSTM,
    'ian': IAN,
    'memnet': MemNet,
    'ram': RAM,
    'cabasc': Cabasc,
    'tnet_lf': TNet_LF,
    'aoa': AOA,
    'mgan': MGAN,
    'bert_spc': BERT_SPC,
    'lcf_bert': LCF_BERT,
    'aen_bert': AEN_BERT
}

inputs_columns = {
    'lstm': ['text_raw_indices'],
    'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
    'atae_lstm': ['text_raw_indices', 'aspect_indices'],
    'ian': ['text_raw_indices', 'aspect_indices'],
    'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
    'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
    'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices',
               'text_right_with_aspect_indices'],
    'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
    'aoa': ['text_raw_indices', 'aspect_indices'],
    'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
    'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
    'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
    'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
}
initializers = {
    'xavier_uniform_': torch.nn.init.xavier_uniform_,
    'xavier_normal_': torch.nn.init.xavier_normal,
    'orthogonal_': torch.nn.init.orthogonal_,
}
optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD,
}
