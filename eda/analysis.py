# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import pandas as pd
import io
import random

from eda import args

if os.path.exists(args.eda_report_path):
    os.system('rm report.txt')


def analyzefile(file):
    print(file)
    df = pd.read_csv(file)
    os.system("touch " + os.path.basename(args.eda_report_path))
    os.system(
        "echo '-------------information about " + os.path.basename(file) + " set------------' >> " + os.path.basename(
            args.eda_report_path))
    os.system(
        "echo 'the row number of " + os.path.basename(file) + " is " + str(df.shape[0]) + "' >> " + os.path.basename(
            args.eda_report_path))
    if file == args.train_100k_path:
        os.system(
            "echo 'the label number of " + os.path.basename(file) + " is\n" + str(
                df[args.output_categories].value_counts()[:10]) + "' >> " + os.path.basename(args.eda_report_path))
    os.system("echo '\n-------------the describe of " + os.path.basename(file) + " is ------------------ \n" + str(
        df.describe()) + "' >> " + os.path.basename(args.eda_report_path))
    os.system("echo '\n--------------the info of " + os.path.basename(
        file) + " data --------------- \n' >> " + os.path.basename(args.eda_report_path))
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    f = open(args.eda_report_path, 'a')
    f.write(info)
    f.write("\n\n\n")
    f.close()


def cal(text):
    df = pd.read_csv(text)
    df0 = len(df[df[args.output_categories] == '-1'])
    print(df0)
    df1 = len(df[df[args.output_categories] == '0'])
    print(df1)
    df2 = len(df[df[args.output_categories] == '1'])
    print(df2)
    sum = df1 + df2 + df0
    print(df0 / sum, df1 / sum, df2 / sum)


def generate_fold_data(train_df, test_df):
    train_df = train_df[train_df[args.output_categories].isin(['-1', '0', '1'])]
    test_df[args.input_categories] = test_df[args.input_categories].fillna('无。')
    train_df[args.input_categories] = train_df[args.input_categories].fillna('无。')
    test_df[args.output_categories] = 0

    if os.path.exists(args.eda_data_dir) is False:
        os.makedirs(args.eda_data_dir)
    test_df.to_csv(args.eda_data_test_10k_path, index=False, encoding='utf-8')
    train_df.to_csv(args.eda_data_train_10k_path, index=False, encoding='utf-8')
    index = set(range(train_df.shape[0]))
    K_fold = []
    for i in range(5):
        if i == 4:
            tmp = index
        else:
            tmp = random.sample(index, int(1.0 / 5 * train_df.shape[0]))
        index = index - set(tmp)
        print("Number:", len(tmp))
        K_fold.append(tmp)

    for i in range(5):
        print("Fold", i)
        if os.path.exists(os.path.join(args.eda_dir, 'data_{}'.format(i))) is False:
            os.makedirs(os.path.join(args.eda_dir, 'data_{}'.format(i)))
        dev_index = list(K_fold[i])
        train_index = []
        for j in range(5):
            if j != i:
                train_index += K_fold[j]
        train_df.iloc[train_index].to_csv(
            os.path.join(args.eda_dir, "data_{}".format(i), args.train_80k_name), index=False,
            encoding='utf-8')
        train_df.iloc[dev_index].to_csv(
            os.path.join(args.eda_dir, "data_{}".format(i), args.dev_20k_name), index=False, encoding='utf-8')
        test_df.to_csv(os.path.join(args.eda_dir, "data_{}".format(i), args.test_10k_name), index=False,
                       encoding='utf-8')


# """
# 分析文件
# """
# analyzefile(args.train_100k_path)
# analyzefile(args.train_900k_path)
# analyzefile(args.test_10k_path)
#
# """
# 分析数据集
# 16902
# 57619
# 25392
# 0.1691671754426351 0.5766917217979642 0.25414110275940066
# """
# cal(args.train_100k_path)

"""
生成5折数据集
Fold 0: DEV Number: 19983
Fold 1: DEV Number: 19983
Fold 2: DEV Number: 19983
Fold 3: DEV Number: 19983
Fold 4: DEV Number: 19987

"""
random.seed(1)
train_df = pd.read_csv(args.train_100k_path)
test_df = pd.read_csv(args.test_10k_path)
generate_fold_data(train_df=train_df, test_df=test_df)
