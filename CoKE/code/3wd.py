"""三支决策"""
import os
import generate_knowledge
import re

# 加载模型
print('generating konwledge model loading...')
comet = generate_knowledge.Comet("/comet-atomic_2020_BART")
comet.model.zero_grad()
print("model loaded")

topK = str(16)

for rel in ["oEffect", "xEffect", "oReact", "xReact", "xIntent", "Causes"]:
    os.makedirs(f"./processed_3wd_" + rel + "/combined_maxsim" + topK, exist_ok=True)
    os.makedirs(f"./processed_3wd_" + rel + "/combined_maxsim" + topK + "/train", exist_ok=True)
    os.makedirs(f"./processed_3wd_" + rel + "/combined_maxsim" + topK + "/test", exist_ok=True)
    for bath_path in ['train', 'test']:
        folder = f"./processed_score/combined_maxsim" + topK + "/" + bath_path
        files = sorted(os.listdir(folder))

        for file in files:
            sel_posts = []
            remain_posts = []
            queries = []
            results = []
            with open(folder + '/' + file, 'r') as f:
                contents = f.readlines()
                # 去除换行符
                contents = [cont.strip() for cont in contents]
                # print(contents)
                for ct in contents:
                    # 使用正则表达式匹配数字
                    match = re.search(r'\d+\.?\d*$', ct)
                    if match:
                        score = float(match.group())
                        post = ct.split(match.group())[0]
                        # 阈值
                        if 0.2 <= score <= 0.5:
                            sel_posts.append(post)
                            query = "{} {} [GEN]".format(post, rel)
                            queries.append(query)
                        else:
                            remain_posts.append(post)
                results = comet.generate(queries, decode_method="beam", num_generate=5)
                for i in  range(len(sel_posts)):
                    for j in results[i]:
                        sel_posts[i] += ' ==sep== ' + j

            with open(f"./processed_3wd_" + rel + "/combined_maxsim"+ topK + "/" + bath_path + "/" + file, 'w') as f:
                for sel in sel_posts:
                    f.write(sel + "\n")
                for rem in remain_posts:
                    f.write(rem + "\n")