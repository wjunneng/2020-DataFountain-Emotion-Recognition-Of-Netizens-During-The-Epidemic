# -*-coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import pandas as pd
import numpy as np
from bert.conf import args
from bert.bin.predict import Predict
from bert.bin.train import Instructor

if __name__ == '__main__':
    # -* train *-
    df_train = pd.read_csv(args.train_100k_path, engine='python', sep=',', encoding='utf-8')
    df_train = df_train[df_train[args.output_categories].isin(['-1', '0', '1'])]

    args.model_class = args.model_classes[args.model_name]
    args.inputs_cols = args.inputs_cols[args.model_name]
    args.initializer = args.initializers[args.initializer]
    args.optimizer = args.optimizers[args.optimizer]

    instructor = Instructor(df_train)
    instructor.main()

    # -* predict *-
    df_test = pd.read_csv(args.test_10k_path, engine='python', sep=',', encoding='utf-8')
    df_sub = pd.read_csv(args.submit_example_path)

    df_sub['y'] = np.asarray(Predict(text=df_test[args.input_categories].values,
                                     target=df_test[args.target_categories].values).predict_all()) - 1
    df_sub['id'] = df_sub['id'].apply(lambda x: str(x) + ' ')
    df_sub.to_csv(args.submit_path, index=False, encoding='utf-8')
