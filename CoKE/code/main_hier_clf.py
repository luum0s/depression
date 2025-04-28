import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from data import KnowHierDataModule 
from model import HierKnowClassifier
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    '''** 用于解包字典或关键字参数，将字典或关键字参数中的键值对作为单独的参数传递给函数或方法。'''
    model = model_type(**vars(args))    #创建了一个HierClassifier模型对象
    '''
    BertTokenizerFast(name_or_path='bert-base-uncased', vocab_size=30522, model_max_length=512, 
    is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 
    'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
    '''
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)  #用于文本处理的工具，用于将文本转换为数字或向量形式以供模型进行预测或训练。
    data_module = KnowHierDataModule(args.bs, args.input_dir, tokenizer, args.max_len) #data_module 是用于加载和处理数据的类
    # data_module = HierDataModule(args.bs, args.input_dir, tokenizer, args.max_len)
    '''
    定义了一个 EarlyStopping 回调函数,用于在训练过程中进行早期停止.
    "max",即当监测指标f1增大时,应停止训练。
    '''
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=args.patience,
        mode="max"
    )
    '''
    是一个回调函数，它可以帮助在训练过程中自动保存最佳模型权重
    它会在每个训练 epoch 结束时，根据 val_f1 指标的表现保存最好的模型权重
    mode 参数指定了最佳模型权重的评估标准，此处设置为 max,表示监测指标越大越好
    '''
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max"
    )
    
    # 创建了一个 PyTorch Lightning 的 Trainer 对象，用于训练和评估模型。
    # 原代码：gpus=1表示使用gpu的数量为1,现在指定不用gpu训练
    '''
    两个回调函数:early_stop_callback:表示提前停止训练以防止过拟合,checkpoint_callback:表示用于保存模型的checkpoint以便在训练过程中恢复训练或在测试阶段使用。
    '''

    #   定义了 PyTorch Lightning 的训练器 trainer，它将执行整个训练过程并监控训练的进度
    trainer = pl.Trainer(gpus=1, callbacks=[early_stop_callback, checkpoint_callback], val_check_interval=1.0, max_epochs=100, min_epochs=1, accumulate_grad_batches=args.accumulation, gradient_clip_val=args.gradient_clip_val, deterministic=True, log_every_n_steps=100)

    if args.find_lr:
            #运行学习率寻找器来寻找最优学习率   
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)

        new_lr = lr_finder.suggestion()
        print(new_lr)

    else:
        # 运行正常的训练过程
        trainer.fit(model, data_module)


if __name__ == "__main__":
    '''
    这段代码是一个命令行解析器，它允许用户从命令行中输入一些参数来运行脚本。
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--hier", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2021)
    temp_args, _ = parser.parse_known_args()    #   函数解析命令行参数，temp_args 包含已知的参数，_ 包含未知的参数
    seed_everything(temp_args.seed, workers=True)   #   函数设置随机种子，以确保实验的可复现性
    model_type = HierKnowClassifier
    # model_type = HierClassifier                     #定义模型类型 model_type 为 HierClassifier，该模型用于进行分类任务
    parser = model_type.add_model_specific_args(parser) #将模型特定的参数添加到命令行解析器中，以便在运行脚本时可以指定这些参数的值。
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--input_dir", type=str, default="/processed/combined_maxsim16")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)
    parser.add_argument("--find_lr", action="store_true")
    
    args = parser.parse_args()
    
    main(args)

    