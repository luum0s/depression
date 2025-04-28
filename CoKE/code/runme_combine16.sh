CUDA_VISIBLE_DEVICES=0 nohup python -u main_hier_clf.py --model_type bert-base-uncased --lr=2e-5 --input_dir ./processed/combined_maxsim16 --bs=4 --user_encoder=trans_know --num_trans_layers=4 > result.txt 2>&1 &

# 这个命令运行名为 main_hier_clf.py 的 Python 脚本，并传递了一些命令行参数。其中：

#     CUDA_VISIBLE_DEVICES=2 是指定使用第二个 GPU 设备运行脚本（如果系统有多个 GPU 的话）；
#     -u 表示让 Python 在无缓冲模式下运行，即实时输出脚本的打印信息；
#     --model_type bert-base-uncased 指定了使用的预训练模型为 bert-base-uncased；
#     --lr=2e-5 指定了学习率为 2e-5；
#     --input_dir ./processed/combined_maxsim16 指定了输入数据所在的文件夹路径，这里可以换成常识知识目录，三支决策目录和知识筛选目录
#     --bs=4 指定了批次大小为 4；
#     --user_encoder=trans_know 指定了用户嵌入的编码方式为 trans_know；
#     --num_trans_layers=4 指定了使用 4 层 Transformer 进行编码。