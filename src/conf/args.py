# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# -* orignal dir/path *-
projec_dir = os.path.dirname(os.path.abspath('..'))
input_dir = os.path.join(projec_dir, 'input')
test_dataset_dir = os.path.join(input_dir, 'test_dataset')
train_dataset_dir = os.path.join(input_dir, 'train_dataset')
pretain_model_dir = os.path.join(input_dir, 'pretain_model')

test_10k_path = os.path.join(test_dataset_dir, 'nCov_10k_test.csv')
train_100k_path = os.path.join(train_dataset_dir, 'nCov_100k_train.labeled.csv')
train_900k_path = os.path.join(train_dataset_dir, 'nCov_900k_train.unlabeled.csv')
submit_example_path = os.path.join(input_dir, 'submit_example.csv')


# -* generate dir/path *-
output_dir = os.path.join(projec_dir, 'output')