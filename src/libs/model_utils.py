# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

os.chdir(sys.path[0])

import collections
import unicodedata
import six
import torch
from torch import nn

SPIECE_UNDERLINE = '▁'

from src.libs import zh_wiki
from src.confs import arguments


class Util(object):
    @staticmethod
    def load_vocab(vocab_file):
        """
            Loads a vocabulary file into a dictionary.
        """
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    @staticmethod
    def whitespace_tokenize(text):
        """
            Runs basic whitespace cleaning and splitting on a peice of text.
        """
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    @staticmethod
    def printable_text(text):
        """
            Returns text encoded in a way suitable for print or `tf.logging`.
        """
        # These functions want `str` for both Python2 and Python3, but in one case
        # it's a Unicode string and in the other it's a byte string.
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text
            elif isinstance(text, str):
                return text.encode("utf-8")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    @staticmethod
    def convert_to_unicode(text):
        """
            Converts `text` to Unicode (if it's not already), assuming utf-8 input.
        """
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, str):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    @staticmethod
    def _is_whitespace(char):
        """
            Checks whether `chars` is a whitespace character.
        """
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r" or ord(char) == 0x202F:
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    @staticmethod
    def _is_control(char):
        """
            Checks whether `chars` is a control character.
        """
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    @staticmethod
    def _is_punctuation(char):
        """
            Checks whether `chars` is a punctuation character.
        """
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    @staticmethod
    def _is_chinese_char(cp):
        """
            Checks whether CP is the codepoint of a CJK character.
        """
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or
                (cp >= 0x20000 and cp <= 0x2A6DF) or (cp >= 0x2A700 and cp <= 0x2B73F) or
                (cp >= 0x2B740 and cp <= 0x2B81F) or (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True

        return False

    @staticmethod
    def _is_fuhao(c):
        """
            Checks whether char is fuhao or not.
        """
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’':
            return True
        return False

    @staticmethod
    def _clean_text(text):
        """
            Performs invalid character removal and whitespace cleanup on text.
        """
        output = []
        for char in text:
            # codepoint
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or Util._is_control(char):
                continue
            if Util._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _tokenize_chinese_chars(text):
        """
            Adds whitespace around any CJK character.
        """
        output = []
        for char in text:
            cp = ord(char)
            if Util._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _run_strip_accents(text):
        """
            Strips accents from a piece of text.
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            # Nonspacing mask
            if cat == "Mn":
                continue
            output.append(char)

        return "".join(output)

    @staticmethod
    def _run_split_on_punc(text):
        """
            Splits punctuation on a piece of text.
        """
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if Util._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    @staticmethod
    def hant_to_hans(text):
        _zh2Hant, _zh2Hans = {}, {}
        for old, new in ((zh_wiki.zh2Hant, _zh2Hant), (zh_wiki.zh2Hans, _zh2Hans)):
            for k, v in old.items():
                new[k] = v

        _zh2Hant = {value: key for (key, value) in _zh2Hant.items() if key != value}
        _zh2Hans = {key: value for (key, value) in _zh2Hans.items() if key != value}

        _zh2Hant.update(_zh2Hans)

        for hant in _zh2Hant.keys():
            text = text.replace(hant, _zh2Hant[hant])

        return text

    @staticmethod
    def deal_text(text):
        """
            Tokenizes a piece of text.
        """
        text = Util.convert_to_unicode(text)
        text = Util._clean_text(text)
        text = Util._tokenize_chinese_chars(text)

        orig_tokens = Util.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            token = Util._run_strip_accents(token)
            split_tokens.extend(Util._run_split_on_punc(token))

        output_tokens = Util.whitespace_tokenize(" ".join(split_tokens))

        return ''.join(output_tokens)


class LabelSmoothingLoss(nn.Module):
    """
    Provides Label-Smoothing loss.

    Args:
        vocab_size (int): the number of classfication
        ignore_index (int): Indexes that are ignored when calculating loss
        smoothing (float): ratio of smoothing (confidence = 1.0 - smoothing)
        dim (int): dimention of calculation loss
        logit (torch.Tensor): probability distribution value from model and it has a logarithm shape
        target (torch.Tensor): ground-thruth encoded to integers which directly point a word in label

    Returns: label_smoothed
        - **label_smoothed** (float): sum of loss

    Reference:
        https://github.com/pytorch/pytorch/issues/7455
    """

    # def __init__(self, categories=arguments.categories, ignore_index=-100, smoothing=0.1, dim=-1):
    #     super(LabelSmoothingLoss, self).__init__()
    #     self.confidence = 1.0 - smoothing
    #     self.smoothing = smoothing
    #     self.categories = categories
    #     self.dim = dim
    #     self.ignore_index = ignore_index
    #
    # def forward(self, logit, target):
    #     with torch.no_grad():
    #         label_smoothed = torch.zeros_like(logit)
    #         label_smoothed.fill_(self.smoothing / (self.categories - 1))
    #         label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
    #         label_smoothed[target == self.ignore_index, :] = 0
    #
    #     return torch.mean(torch.sum(-label_smoothed * logit))

    def __init__(self, classes=arguments.categories, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
