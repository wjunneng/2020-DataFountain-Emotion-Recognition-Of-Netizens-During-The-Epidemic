# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# -* orignal dir/path *-
# project_dir = os.path.dirname(os.path.abspath('..'))
project_dir = '/home/wjunneng/Ubuntu/2020-DataFountain-Emotion-Recognition-Of-Netizens-During-The-Epidemic'
# project_dir = '/content/2020-DataFountain-Emotion-Recognition-Of-Netizens-During-The-Epidemic'

data_dir = os.path.join(project_dir, 'data')
input_dir = os.path.join(data_dir, 'input')
test_dataset_dir = os.path.join(input_dir, 'test_dataset')
train_dataset_dir = os.path.join(input_dir, 'train_dataset')

# -* test *-
test_10k_path = os.path.join(test_dataset_dir, 'nCov_10k_test_utf_8.csv')
train_100k_path = os.path.join(train_dataset_dir, 'nCoV_100k_train_utf_8.labled.csv')
train_900k_path = os.path.join(train_dataset_dir, 'nCoV_900k_train_utf_8.unlabled.csv')

# -* output *-
fold_dir = os.path.join(data_dir, 'fold')
fold_report_path = os.path.join(fold_dir, 'report.txt')
fold_data_dir = os.path.join(fold_dir, 'data')
fold_data_train_10k_path = os.path.join(fold_data_dir, 'nCoV_100k_train_utf_8.labled.csv')
fold_data_test_10k_path = os.path.join(fold_data_dir, 'nCov_10k_test_utf_8.csv')

submit_example_path = os.path.join(input_dir, 'submit_example.csv')
fold_submission_path = os.path.join(fold_data_dir, 'submission.csv')
fold_data_0_sub_path = os.path.join(fold_dir, 'data_0/sub.csv')
fold_data_1_sub_path = os.path.join(fold_dir, 'data_1/sub.csv')
fold_data_2_sub_path = os.path.join(fold_dir, 'data_2/sub.csv')
fold_data_3_sub_path = os.path.join(fold_dir, 'data_3/sub.csv')
fold_data_4_sub_path = os.path.join(fold_dir, 'data_4/sub.csv')

input_categories = '微博中文内容'
output_categories = '情感倾向'
categories = 3
weibo_id = '微博id'
content = '微博中文内容'
title = '发布人账号'
flag = '情感倾向'
id = 'id'
y = 'y'

label_0 = 'label_0'
label_1 = 'label_1'
label_2 = 'label_2'

test_10k_name = 'nCov_10k_test_utf_8.csv'
dev_20k_name = 'nCoV_20k_dev_utf_8.labled.csv'
train_80k_name = 'nCoV_80k_train_utf_8.labled.csv'

# test_10k_name = 'test.csv'
# dev_20k_name = 'dev.csv'
# train_80k_name = 'train.csv'
