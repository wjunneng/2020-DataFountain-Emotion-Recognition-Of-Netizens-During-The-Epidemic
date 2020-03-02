# -*-coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import math
import torch
import random
import logging
import numpy as np
import pandas as pd
from time import strftime, localtime
from pytorch_transformers import BertModel
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import StratifiedKFold

from bert.conf import args
from bert.lib.data_util import Tokenizer4Bert, ABSADataset, Util, LabelSmoothingLoss, PreProcessing

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Instructor(object):
    """
    特点：使用flyai字典的get all data  | 自己进行划分next batch
    """

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, df_sub: pd.DataFrame):
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if os.path.exists(args.log_dir) is False:
            os.mkdir(args.log_dir)
        log_file = '{}-{}.log'.format(args.model_name, strftime('%y%m%d-%H%M', localtime()))
        logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, log_file)))

        self.args = args
        self.df_train = df_train

        if 'bert' in self.args.model_name:
            self.tokenizer = Tokenizer4Bert(max_seq_len=self.args.max_seq_len,
                                            pretrained_bert_name=self.args.pretain_model_dir)
            bert = BertModel.from_pretrained(pretrained_model_name_or_path=self.args.pretain_model_dir)
            self.model = self.args.model_classes[self.args.model_name](bert, self.args).to(DEVICE)

        if DEVICE.type == 'cuda':
            logger.info(
                'cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=DEVICE.index)))

        Util.print_args(model=self.model, logger=logger, args=self.args)

        self.train_target = df_train[self.args.target_categories].values
        self.train_text = df_train[self.args.input_categories].values
        self.train_stance = df_train[self.args.output_categories].values

        self.test_text = df_test[args.input_categories].values
        self.test_stance = df_test[args.target_categories].values

        self.df_sub = df_sub

    def main(self):
        # loss and optimizer
        # criterion = nn.CrossEntropyLoss()
        # 标签平滑
        criterion = LabelSmoothingLoss(classes=len(self.args.labels), smoothing=0.2)
        # Focal Loss
        # criterion = FocalLoss(class_num=len(self.args.labels))

        _params = filter(lambda x: x.requires_grad, self.model.parameters())
        optimizer = self.args.optimizer(_params, lr=self.args.learning_rate,
                                        weight_decay=self.args.l2reg)

        # 训练
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        best_model_path = None
        Util.reset_params(model=self.model, args=self.args)

        for epoch in range(self.args.EPOCHS):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()

            test_preds = []
            for fold, (train_index, valid_index) in enumerate(
                    StratifiedKFold(n_splits=5).split(X=self.train_text, y=self.train_stance,
                                                      groups=self.train_target)):
                logger.info('>' * 20)
                logger.info('fold: {}'.format(fold))
                train_text_fold = self.train_text[train_index]
                train_stance_fold = self.train_stance[train_index].astype(str)
                train_target_fold = self.train_target[train_index]

                valid_text_fold = self.train_text[valid_index]
                valid_stance_fold = self.train_stance[valid_index].astype(str)
                valid_target_fold = self.train_target[valid_index]

                print('train.shape: {}'.format(train_text_fold.shape))
                print('valid.shape: {}'.format(valid_text_fold.shape))

                self.trainset = ABSADataset(data_type=None,
                                            fname=(train_target_fold, train_text_fold, train_stance_fold),
                                            tokenizer=self.tokenizer)

                self.validset = ABSADataset(data_type=None,
                                            fname=(valid_target_fold, valid_text_fold, valid_stance_fold),
                                            tokenizer=self.tokenizer)

                train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.args.BATCH, shuffle=True)
                val_data_loader = DataLoader(dataset=self.validset, batch_size=self.args.BATCH, shuffle=False)

                print('step: {}'.format(len(train_data_loader) // self.args.BATCH))
                for i_batch, sample_batched in enumerate(train_data_loader):
                    global_step += 1
                    optimizer.zero_grad()

                    inputs = [sample_batched[col].to(DEVICE) for col in self.args.inputs_columns[self.args.model_name]]
                    outputs = self.model(inputs)
                    targets = torch.tensor(sample_batched['polarity']).to(DEVICE)

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    loss_total += loss.item() * len(outputs)
                    if global_step % self.args.log_step == 0:
                        train_acc = n_correct / n_total
                        train_loss = loss_total / n_total
                        logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

                self.model.eval()
                val_acc, val_f1 = Util.evaluate_acc_f1(model=self.model, args=self.args, data_loader=val_data_loader)
                logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
                if val_acc > max_val_acc and not self.args.do_fold:
                    max_val_acc = val_acc
                    best_model_path = self.args.output_dir
                    Util.save_model(model=self.model, output_dir=best_model_path)
                    logger.info('>> saved: {}'.format(best_model_path))
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1

                test_batch_preds = None
                for i in range(math.ceil(len(self.test_text) / self.args.BATCH)):
                    batch_text_data = [self.test_text[i] for i in
                                       range(i * self.args.BATCH, min((i + 1) * self.args.BATCH, len(self.test_text)))]
                    batch_target_data = [self.test_stance[i] for i in
                                         range(i * self.args.BATCH,
                                               min((i + 1) * self.args.BATCH, len(self.test_stance)))]
                    TARGET = [i for i in batch_target_data]
                    TEXT = [i for i in batch_text_data]

                    self.predict_loader = DataLoader(
                        dataset=ABSADataset(data_type=None, fname=(TARGET, TEXT, None), tokenizer=self.tokenizer),
                        batch_size=self.args.BATCH)

                    outputs = None
                    for i_batch, sample_batched in enumerate(self.predict_loader):
                        inputs = [sample_batched[col].to(DEVICE) for col in
                                  self.args.inputs_columns[self.args.model_name]]
                        outputs = self.model(inputs)

                    if test_batch_preds is None:
                        test_batch_preds = outputs.data.cpu().numpy()
                    else:
                        test_batch_preds = np.vstack((test_batch_preds, outputs.data.cpu().numpy()))

                test_preds.append(test_batch_preds)

            sub = np.average(test_preds, axis=0)
            sub = np.argmax(sub, axis=1)
            self.df_sub['y'] = sub - 1
            self.df_sub['id'] = self.df_sub['id'].apply(lambda x: str(x))
            self.df_sub.to_csv(args.submit_path, index=False, encoding='utf-8')

        logger.info('> max_val_acc: {0} max_val_f1: {1}'.format(max_val_acc, max_val_f1))
        logger.info('> train save model path: {}'.format(best_model_path))
