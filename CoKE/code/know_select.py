"""知识筛选"""
import string
import os

topK = str(16)

class SentiWordNet():
    def __init__(self, netpath):
        self.netpath = netpath
        self.dictionary = {}

    def infoextract(self):
        try:
            f = open(self.netpath, "r")
        except IOError:
            print("failed to open file!")
            exit()
        print('start extracting.......')

        # Example line:
        # POS     ID     PosS  NegS SynsetTerm#sensenumber Desc
        # a   00009618  0.5    0.25  spartan#4 austere#3 ascetical#2  ……
        for sor in f.readlines():
            if sor.strip().startswith("#"):
                pass
            else:
                data = sor.split("\t")
                if len(data) != 6:
                    print('invalid data')
                    break
                synsetScore = float(data[2]) - float(data[3])  # // Calculate synset score as score = PosS - NegS
                synTermsSplit = data[4].split(" ")  # word#id
                # ["dorsal#2", "abaxial#1"]
                for w in synTermsSplit:
                    synTermAndRank = w.split("#")
                    synTerm = synTermAndRank[0]
                    self.dictionary[synTerm] = self.dictionary.get(synTerm, 0) + synsetScore

    def getscore(self, word):
        return self.dictionary.get(word, 0)


def get_major_sent_know():
    netpath = "/models/SentiWordNet_3.0.0.txt"
    for way in ["know", "3wd"]:
        for relation in ["oEffect", "xEffect", "oReact", "xReact", "xIntent", "Causes"]:
            # 创建目录
            os.makedirs(f"./processed_ext_sel_" + way + "_" + relation + "/combined_maxsim" + topK, exist_ok=True)
            os.makedirs(f"./processed_ext_sel_" + way + "_" + relation + "/combined_maxsim" + topK + "/train", exist_ok=True)
            os.makedirs(f"./processed_ext_sel_" + way + "_" + relation + "/combined_maxsim" + topK + "/test", exist_ok=True)
            # 读取输入文件
            for bath_path in ['train', 'test']:
                folder = f"./processed_" + way + "_" + relation + "/combined_maxsim" + topK + "/"  + bath_path
                files = sorted(os.listdir(folder))
                for file in files:

                    swn = SentiWordNet(netpath)
                    swn.infoextract()

                    with open(os.path.join(folder, file), 'r') as f:
                        sep = " ==sep== "
                        res = []
                        for num, sent in enumerate(f.readlines()):
                            temp = sent.split(" ==sep== ")
                            id_know = {}
                            id_score = {}
                            total_score = 0
                            for i, know in enumerate(temp[1:]):
                                id_know[i] = know.strip()
                                id_score[i] = 0
                                for word in id_know[i].split():
                                    id_score[i] += float(swn.getscore(word))
                                total_score += id_score[i]
                            content = []
                            if total_score == 0:
                                for key in id_know.keys():
                                    content.append(id_know[key])
                            elif total_score > 0:
                                for key in id_know.keys():
                                    if id_score[key] >= 0:
                                        content.append(id_know[key])
                            else:
                                for key in id_know.keys():
                                    if id_score[key] <= 0:
                                        content.append(id_know[key])
                            if len(content) == 0:
                                temp[0] =  temp[0].rstrip('\n')
                                res_record = temp[0]
                            else:
                                res_record = temp[0] + sep + sep.join(content)
                            res.append(res_record)
                            
                        with open(f"./processed_ext_sel_" + way + "_" + relation + "/combined_maxsim" + topK + "/"  + bath_path + '/' + file, 'w') as f:
                            for r in res:
                                f.write(r + "\n")

get_major_sent_know()