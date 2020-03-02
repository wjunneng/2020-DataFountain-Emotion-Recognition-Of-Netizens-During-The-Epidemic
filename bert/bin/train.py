# -*-coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import torch
import random
import logging
import numpy as np
import pandas as pd
from time import strftime, localtime
from pytorch_transformers import BertModel
from torch.utils.data import DataLoader, random_split

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

    def __init__(self, df_train: pd.DataFrame):
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

        train_target = df_train[self.args.target_categories].values
        train_text = df_train[self.args.input_categories].values
        train_stance = df_train[self.args.output_categories].values

        # ############################# 特征词库的方法 效果不好
        # train_data = pd.DataFrame(data=[stance, target, text]).T
        # train_data.columns = ['STANCE', 'TARGET', 'TEXT']
        # Util.calculate_word_count(train_data)
        # ############################# 特征词库的方法 效果不好

        self.target_set = set()
        for tar in train_target:
            self.target_set.add(tar)
        # train_text = PreProcessing(train_text).get_file_text()

        # ############################# 同义词替换的方法 效果不好
        # self.synonyms = SynonymsReplacer()
        # text_add = []
        # for index in range(len(text)):
        #     text_add.append(self.synonyms.get_syno_sents_list(text[index]))
        # target = np.append(target, target)
        # text = np.append(text, np.asarray(text_add))
        # stance = np.append(stance, stance)
        # ############################# 同义词替换的方法 效果不好

        # print('target.shape: {}, text.shape: {}, stance.shape: {}'.format(target.shape, text.shape, stance.shape))
        trainset = ABSADataset(data_type=None, fname=(train_target, train_text, train_stance), tokenizer=self.tokenizer)

        valset_len = int(len(trainset) * self.args.valset_ratio)
        self.trainset, self.valset = random_split(trainset, (len(trainset) - valset_len, valset_len))

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

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.args.BATCH, shuffle=True)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.args.BATCH, shuffle=False)

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
            print('step: {}'.format(len(train_data_loader) // self.args.BATCH))
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(DEVICE) for col in self.args.inputs_columns]
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
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                best_model_path = self.args.output_dir
                Util.save_model(model=self.model, output_dir=best_model_path)
                logger.info('>> saved: {}'.format(best_model_path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        logger.info('> max_val_acc: {0} max_val_f1: {1}'.format(max_val_acc, max_val_f1))
        logger.info('> train save model path: {}'.format(best_model_path))
