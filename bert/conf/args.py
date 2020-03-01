# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# -* orignal dir/path *-
projec_dir = os.path.dirname(os.path.abspath('..'))
data_dir = os.path.join(projec_dir, 'data')
input_dir = os.path.join(data_dir, 'input')
test_dataset_dir = os.path.join(input_dir, 'test_dataset')
train_dataset_dir = os.path.join(input_dir, 'train_dataset')
pretain_model_dir = os.path.join(data_dir, 'pretain_model')

test_10k_path = os.path.join(test_dataset_dir, 'nCov_10k_test_utf_8.csv')
train_100k_path = os.path.join(train_dataset_dir, 'nCoV_100k_train_utf_8.labled.csv')
train_900k_path = os.path.join(train_dataset_dir, 'nCoV_900k_train_utf_8.unlabled.csv')
submit_example_path = os.path.join(input_dir, 'submit_example.csv')

# -* generate dir/path *-
output_dir = os.path.join(projec_dir, 'output')

output_categories = '情感倾向'
