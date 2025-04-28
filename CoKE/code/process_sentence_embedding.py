# %%
import os
import re
import random
import json
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import xml.dom.minidom
import string
from tqdm import tqdm 
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
import nltk
from nltk.corpus import stopwords
from natsort import natsorted


# %%
sbert = SentenceTransformer('./paraphrase-MiniLM-L6-v2') #sentence-transformers模型


# %%
'''
定义了一个名为"get_input_data"的函数，它接受文件路径和文件名作为输入，并返回文件中的文本数据和帖子数量。
'''
def get_input_data(file, path):
    post_num = 0
    posts = []
    for i in range(10):
        dom = xml.dom.minidom.parse(path + "/" + file+str(i+1)+".xml")
        collection = dom.documentElement
        title = collection.getElementsByTagName('TITLE')
        text = collection.getElementsByTagName('TEXT')
        for i in range(len(title)):
            post = title[i].firstChild.data + ' ' + text[i].firstChild.data
            post = re.sub('\n', ' ', post)
            if len(post) > 0:
                posts.append(post.strip())
                post_num = post_num + 1
    return posts, post_num

# %%
train_posts = []
train_tags = []
train_mappings = []
test_posts = []
test_tags = []
test_mappings = []  
for base_path in ["negative_examples_anonymous", "negative_examples_test", "positive_examples_anonymous", "positive_examples_test"]:
    base_path = "./dataset/"+base_path

    filenames = sorted(os.listdir(base_path)) # 对base_path(一个子数据集)下的文件列表进行排序并返回

    # 处理10个xml连接起来
    for i in range(len(filenames)):
        filenames[i] = filenames[i][:filenames[i].rfind("_")+1]
    filenames=list(set(filenames))

    for fname in filenames: 
        posts, post_num = get_input_data(fname, base_path) #fname是文件名，base_bath是该文件前缀路径
        if "anonymous" in base_path:
            # 训练集
            train_mappings.append(list(range(len(train_posts), len(train_posts)+post_num))) # 映射一个文件中所有帖子的位置
            train_posts.extend(posts) # extend()用于在列表尾部一次性追加另一个序列中的多个值
            train_tags.append(int("positive" in base_path)) # 记录训练集中阳性(帖子)文件的标签,值为"1"
        else:
            # 测试集
            test_mappings.append(list(range(len(test_posts), len(test_posts)+post_num)))
            test_posts.extend(posts)
            test_tags.append(int("positive" in base_path))

# %%
'''
使用"sbert"模型对训练集和测试集中的所有帖子进行编码，将编码的结果存储到变量"train_embs"和"test_embs"中。
'''
train_embs = sbert.encode(train_posts, convert_to_tensor=False)
test_embs = sbert.encode(test_posts, convert_to_tensor=False)

'''
将包含训练集和测试集的字典对象序列化为二进制对象，并存储到文件中。
'''
os.makedirs(f"./processed", exist_ok=True)
with open("./processed/miniLM_L6_embs.pkl", "wb") as f:
    pickle.dump({                                 # pickle.dump(): 将对象序列化成二进制对象；
        "train_posts": train_posts,               # 这里是将包含训练集和测试集的字典对象序列化为二进制对象，并存储到文件中。
        "train_mappings": train_mappings,
        "train_labels": train_tags,
        "train_embs": train_embs,
        "test_posts": test_posts,
        "test_mappings": test_mappings,
        "test_labels": test_tags,
        "test_embs": test_embs,
    }, f)