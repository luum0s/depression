# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
import random

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

logger = logging.getLogger(__name__)
csv.field_size_limit(2147483647)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, label):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputExampleMaskedLM(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, diseaseName):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.diseaseName = diseaseName


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, diseaseName_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.diseaseName_id = diseaseName_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            contents=f.readlines()
            lines0=[]
            for i in contents:
                aa=i.strip()
                a=aa.split('¦')
                lines0.append(a)
            '''
            reader = csv.reader(f, delimiter="¦")
            lines = []
            for line in reader:
                lines.append(line)
            '''
            return lines0


class BinaryProcessor(DataProcessor):
    """Processor for the binary data sets"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_pretrain_examples(self, data_dir):
        return self._create_examples_pretrain(
            self._read_tsv(os.path.join(data_dir, "pretrainData_masked100_preprocessed_onlyNosiy.txt")), "pretrain")

    ####以下函数是处理语料库的入口，待修改
    def get_maskedLM_examples(self, data_dir):
        return self._create_examples_maskedLM(
            self._read_tsv(os.path.join(data_dir, "depressive disorders.txt")), "pretrain")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples_maskedLM(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            '''
            text_a = line[2]
            text_b = line[3] 
            diseaseName = line[0]
            questionType = line[1]
            '''
            text_a = line[0]
            text_b = line[1]
            subtitle_l = line[0].split('| ')
            for i in range(len(subtitle_l)):
                subtitle_l[i] = subtitle_l[i].replace('|', '')
                subtitle_l[i] = subtitle_l[i].strip()
            #对subtitle_l中的元素进行随机抽取（奇数项或者偶数项），多个epoch后可覆盖subtitle_l中的所有元素
            item_masked=[]
            random_n=random.randint(0,1)#生成0,1之间的随机整数
            for i in range(len(subtitle_l)):
                if len(subtitle_l)==1:
                    item_masked.append(subtitle_l[i])
                elif (i+random_n)%2==1:
                    item_masked.append(subtitle_l[i])
            #尝试把text_b的主旨句的医学命名实体进行mask,也许对实验有帮助，仔细设计
            # if label=='2':
            #     label = '1'
            # if random.random()<0.01:
            examples.append(
                InputExampleMaskedLM(guid=guid, text_a=text_a, text_b=text_b, diseaseName=item_masked))
        return examples

    def _create_examples_pretrain(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            # if label=='2':
            #     label = '1'
            # if random.random()<0.1:
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[0]
            # if label=='2':
            #     label = '1'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_example_to_feature(example_row, pad_token=0,
                               sequence_a_segment_id=0, sequence_b_segment_id=1,
                               cls_token_segment_id=1, pad_token_segment_id=0,
                               mask_padding_with_zero=True, sep_token_extra=False):
    example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token, cls_token_segment_id, pad_on_left, pad_token_segment_id, sep_token_extra = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = tokenizer.tokenize(example.text_b)

    tokens_diseaseName=[]
    for i in example.diseaseName:
        tokens_i=tokenizer.tokenize(i)
        tokens_diseaseName+=tokens_i
    #tokens_diseaseName = tokenizer.tokenize(example.diseaseName)

    ids_diseaseName = tokenizer.convert_tokens_to_ids(tokens_diseaseName)
    '''
    for i in range(len(examples)):
        diseaseName=examples[i].diseaseName[0]
        tokens_i=tokenizer.tokenize(diseaseName)
        ids_diseaseName = tokenizer.convert_tokens_to_ids(tokens_i)
        if len(ids_diseaseName) > 35:
            print(i, diseaseName)
    '''
    if len(ids_diseaseName) >40:
        print(i, example.text_a)

    while len(ids_diseaseName) < 40:
        #print('example.diseaseName',example.diseaseName)
        ids_diseaseName.append(0)

    #tokens_questionType = tokenizer.tokenize(example.questionType)

    # tokens_a = tokens_questionType # +

    masked_labels = []  # this -100 is for CLS
    # maskTokenId = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    for tokenIndex in range(len(tokens_a)):
        token = tokens_a[tokenIndex]
        if token in tokens_diseaseName:  # or
            tokens_a[tokenIndex] = '[MASK]'
            masked_labels.append(tokenizer.convert_tokens_to_ids(token))
        else:
            masked_labels.append(-100)
    # example.text_b = False
    if example.text_b:
        #tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        special_tokens_count = 4 if sep_token_extra else 3
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
    else:
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    # masked_labels = [-100]*len(tokens)
    masked_labels.append(-100)

    # We use new token [blank] to replace the disease tokens that may appear in the passage.
    # Our core idea is to BERT infer the disease and aspect from a passage.
    # If the ground-truth disease tokens appear in the passage, they will make it much easier
    # for BERT to infer the disease from the passage and lower the performance.
    masked_labels_b = []
    for tokenIndex in range(len(tokens_b)):
        token = tokens_b[tokenIndex]
        if token in tokens_diseaseName:
            tokens_b[tokenIndex] = '[blank]'
            masked_labels_b.append(tokenizer.convert_tokens_to_ids(token))
        else:
            masked_labels_b.append(-100)
    masked_labels_b.append(-100)

    # tokens_b = False
    if tokens_b:
        tokens += tokens_b + [sep_token]
        segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        # masked_labels = masked_labels + [-100]*(len(tokens_b) + 1)
        masked_labels += masked_labels_b

    if cls_token_at_end:
        tokens = tokens + [cls_token]
        segment_ids = segment_ids + [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        masked_labels = [-100] + masked_labels

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    masked_labels = masked_labels + [-100] * padding_length
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(masked_labels) == max_seq_length

    # if output_mode == "classification":
    #     label_id = label_map[example.label]
    # elif output_mode == "regression":
    #     label_id = float(example.label)
    # else:
    #     raise KeyError(output_mode)
    # print(tokens)
    # print(input_ids)
    # print(masked_labels)
    # print(input_mask)
    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=masked_labels,
                         diseaseName_id=ids_diseaseName)


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, sep_token_extra=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 process_count=cpu_count() - 2):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    examples = [(example, label_map, max_seq_length, tokenizer, output_mode, cls_token_at_end, cls_token, sep_token,
                 cls_token_segment_id, pad_on_left, pad_token_segment_id, sep_token_extra) for example in examples]

    with Pool(process_count) as p:
        features = list(tqdm(p.imap(convert_example_to_feature, examples, chunksize=500), total=len(examples)))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        elif len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "binary": BinaryProcessor,
    "regression": BinaryProcessor
}

output_modes = {
    "binary": "classification",
    "regression": "regression"
}