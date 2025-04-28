"""生成常识知识"""
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
    os.makedirs(f"./processed_know_" + rel + "/combined_maxsim" + topK, exist_ok=True)
    os.makedirs(f"./processed_know_" + rel + "/combined_maxsim" + topK + "/train", exist_ok=True)
    os.makedirs(f"./processed_know_" + rel + "/combined_maxsim" + topK + "/test", exist_ok=True)
    for bath_path in ['train', 'test']:
        folder = f"./processed_score/combined_maxsim" + topK + "/" + bath_path
        files = sorted(os.listdir(folder))

        for file in files:
            posts = []
            queries = []
            results = []
            with open(os.path.join(folder, file), 'r') as f:
                contents = f.readlines()
                contents = [cont.strip() for cont in contents]

                for ct in contents:
                    match = re.search(r'\d+\.?\d*$', ct)
                    if match:
                        score = float(match.group())
                        post = ct.split(match.group())[0]
                        posts.append(post)
                        # 构建查询
                        query = "{} {} [GEN]".format(post, rel)
                        queries.append(query)
                        # 生成知识
                results = comet.generate(queries, decode_method="beam", num_generate=5)    
                for i in  range(len(posts)):
                    for j in results[i]:
                        posts[i] += ' ==sep== ' + j

            with open(f"./processed_know_" + rel + "/combined_maxsim"+ topK + "/" + bath_path + "/" + file, "w") as f:
                for p in posts:
                    f.write(p + "\n")