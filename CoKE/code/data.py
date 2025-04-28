from collections import defaultdict
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy.io import loadmat
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from collections import defaultdict

'''
这段代码主要是用PyTorch Lightning实现了两个数据集类和对应的数据模块类。
'''

class FlatDataset(Dataset):
    # 将文本文件加载并转换为模型输入所需的格式
    def __init__(self, input_dir, tokenizer, max_len, split="train"):
        # 初始化,定义输入目录、tokenizer、最大长度、数据分割
        assert split in {"train", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split)
        for fname in os.listdir(input_dir2):
            # 遍历分割目录下的所有文件
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"  从文件名中解析出label
            sample = {}
            sample["text"] = open(os.path.join(input_dir2, fname), encoding="utf-8").read()
            # 读取文本内容
            tokenized = tokenizer(sample["text"], truncation=True, padding='max_length', max_length=max_len)
            # tokenizer处理
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            self.labels.append(label)

    def __len__(self) -> int:
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, index: int):
        # 获取一个样本的数据
        return self.data[index], self.labels[index]

def my_collate_flat(data):
    # 将一个batch的文本数据转换为模型训练所需的张量格式
    labels = []
    processed_batch = defaultdict(list)
    for item, label in data:
        # 遍历一个batch
        for k, v in item.items():
            processed_batch[k].append(v)
        labels.append(label)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        processed_batch[k] = torch.LongTensor(processed_batch[k])
    labels = torch.FloatTensor(labels)
    return processed_batch, labels

class FlatDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def setup(self, stage):#根据训练阶段(fit/test)设置训练集和测试集
        if stage == "fit":
            self.train_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "train")#初始化训练集
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")#初始化测试集
        elif stage == "test":
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")#仅初始化测试集

    def train_dataloader(self):#定义训练阶段的数据加载器
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_flat, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):#定义验证阶段的数据加载器
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_flat, pin_memory=True, num_workers=4)


# 分层
class KnowHierDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", max_posts=64):
        assert split in {"train", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = []
        self.labels = []
        self.filenames = []  # 存储文件名
        input_dir2 = os.path.join(input_dir, split)
        for fname in tqdm(os.listdir(input_dir2)):
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            sample = {}
            with open(os.path.join(input_dir2, fname), encoding="utf-8") as file:
                content = file.read().strip().split("\n")[:max_posts]

            # 分隔帖子和知识
            post_list = []
            knowledge_list = []
            for post in content:
                post_content, *knowledge = post.split("==sep==")  # 解构赋值，将连接后的字符串分隔成帖子和知识
                post_list.append(post_content)
                knowledge_list.append(knowledge)

            # Tokenize posts
            tokenized_posts = [tokenizer(post, truncation=True, padding='max_length', max_length=max_len) for post in post_list]
            sample['posts'] = [{'input_ids': torch.tensor(tokenized_post['input_ids']),
                                'attention_mask': torch.tensor(tokenized_post['attention_mask'])} for tokenized_post in tokenized_posts]
            # sample['posts'] = [{k: torch.tensor(v) for k, v in tokenized_post.items()} for tokenized_post in tokenized_posts]

            # Tokenize knowledge
            # 合并所有知识，形成1个列表
            result_know = [''.join(lst).strip() for lst in knowledge_list]
            # 将知识合并后送入tokenizer
            tokenized_knowledges = [tokenizer(know, truncation=True, padding='max_length', max_length=max_len) for know in result_know]
            # 将知识放入 sample 字典
            sample['knowledges'] = [{'input_ids': torch.tensor(tokenized_knowledge['input_ids']),
                                'attention_mask': torch.tensor(tokenized_knowledge['attention_mask'])} for tokenized_knowledge in tokenized_knowledges]
            

            # for i, knowledge in enumerate(knowledge_list):
            #     tokenized_knowledge = [tokenizer(k, truncation=True, padding='max_length', max_length=max_len) for k in knowledge]
            #     sample[f'knowledge_{i}'] = [{'input_ids': torch.tensor(tokenized_k['input_ids']),
            #                                  'attention_mask': torch.tensor(tokenized_k['attention_mask'])} for tokenized_k in tokenized_knowledge]
                # sample[f'knowledge_{i}'] = [{k: torch.tensor(v) for k, v in tokenized_k.items()} for tokenized_k in tokenized_knowledge]

            self.data.append(sample)
            self.labels.append(label)
            self.filenames.append(fname)  # 保存文件名


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.filenames[index] # 返回文件名

def my_collate_hier(data):
    # Custom collate_fn for handling hierarchical data
    labels = []
    filenames = []
    processed_batch = []
    # for item, label in data:
    for item, label, filename in data:
        user_feats = {}
        # for key, value in item.items():
        #     if key == 'posts':
        #         # Handle posts which is a list of dictionaries
        #         user_feats[key] = [{k: torch.LongTensor(v) for k, v in post.items()} for post in value]
        #     elif key.startswith('knowledge_'):
        #         # Handle knowledge_X which is a list of dictionaries
        #         user_feats[key] = [{k: torch.LongTensor(v) for k, v in knowledge_item.items()} for knowledge_item in value]
        #     else:
        #         # Handle other keys
        #         user_feats[key] = {k: torch.LongTensor(v) for k, v in value.items()}  # Convert sequences to Tensor

        for key, value in item.items():
            if key == 'posts':
                # Handle posts which is a list of dictionaries
                user_feats[key] = [{'input_ids': torch.LongTensor(post['input_ids']),
                                    'attention_mask': torch.LongTensor(post['attention_mask'])} for post in value]
            elif key.startswith('knowledges'):
                # Handle knowledge_X which is a list of dictionaries
                user_feats[key] = [{'input_ids': torch.LongTensor(knowledge_item['input_ids']),
                                    'attention_mask': torch.LongTensor(knowledge_item['attention_mask'])} for knowledge_item in value]
            else:
                # Handle other keys
                user_feats[key] = {'input_ids': torch.LongTensor(value['input_ids']),
                                  'attention_mask': torch.LongTensor(value['attention_mask'])}  # Convert sequences to Tensor
                
        processed_batch.append(user_feats)
        labels.append(label)
        filenames.append(filename)  # 保存文件名

    labels = torch.FloatTensor(np.array(labels))  # Convert labels to Tensor
    return processed_batch, labels, filenames  # 返回文件名列表


class KnowHierDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

    def setup(self, stage):
        if stage == "fit":
            self.train_set = KnowHierDataset(self.input_dir, self.tokenizer, self.max_len, "train")
            self.test_set = KnowHierDataset(self.input_dir, self.tokenizer, self.max_len, "test")
        elif stage == "test":
            self.test_set = KnowHierDataset(self.input_dir, self.tokenizer, self.max_len, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_hier, shuffle=True, pin_memory=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=4)

# Preprocess a single batch of text
def infer_preprocess(tokenizer, texts, max_len):
    batch = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        batch[k] = torch.LongTensor(batch[k])
    return batch


