# -*-coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import torch
import math
import numpy as np
import pandas as pd
from pytorch_transformers import BertModel
from torch.utils.data import DataLoader

from bert.lib.data_util import Util, ABSADataset, Tokenizer4Bert
from bert.conf import args

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Predict(object):
    def __init__(self, text, target):
        self.net = None
        self.text = text
        self.target = target
        self.args = args
        self.idx2label = dict((i, args.labels[i]) for i in range(len(args.labels)))

        self.tokenizer = Tokenizer4Bert(max_seq_len=self.args.max_seq_len,
                                        pretrained_bert_name=self.args.pretain_model_dir)
        bert = BertModel.from_pretrained(pretrained_model_name_or_path=self.args.pretain_model_dir)
        self.model = self.args.model_classes[self.args.model_name](bert, self.args).to(DEVICE)

        self.net = Util.load_model(model=self.model, output_dir=os.path.join(os.getcwd(), self.args.output_dir))

    def do_predict(self, TEXT, TARGET):
        predict_set = ABSADataset(data_type=None, fname=(TARGET.tolist(), TEXT.tolist(), None),
                                  tokenizer=self.tokenizer)
        predict_loader = DataLoader(dataset=predict_set, batch_size=len(TEXT))
        outputs = None
        for i_batch, sample_batched in enumerate(predict_loader):
            inputs = [sample_batched[col].to(DEVICE) for col in self.args.inputs_cols[self.args.model_name]]
            outputs = self.net(inputs)

        outputs = torch.argmax(outputs, dim=-1).cpu().numpy().tolist()

        return outputs

    def predict_all(self):
        """
        预测所有的数据
        :param datas:
        :return:
        """
        labels = []

        for i in range(math.ceil(len(self.text) / self.args.BATCH)):
            batch_text_data = [self.text[i] for i in
                               range(i * self.args.BATCH, min((i + 1) * self.args.BATCH, len(self.text)))]
            batch_target_data = [self.target[i] for i in
                                 range(i * self.args.BATCH, min((i + 1) * self.args.BATCH, len(self.target)))]
            TARGET = [i for i in batch_target_data]
            TEXT = [i for i in batch_text_data]
            labels.extend(self.do_predict(TEXT=np.asarray(TEXT), TARGET=np.asarray(TARGET)))

        return labels


