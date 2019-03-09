import numpy as np
import re
import os
import jieba
from gensim.models.doc2vec import Doc2Vec

if __name__ == '__main__':
    model = Doc2Vec.load(r'D:\PycharmProjects\KeywordExtraction\data\model\sen2vec\patent\bxkdoc_100_dm_40_5.model')
    word_list = list(jieba.cut(r'本发明公开了带自动控制系统的蒸发冷却‑机械制冷联合空调机组，包括有机组壳体，机组壳体两相对的侧壁上分别设置有一次风进风口、送风口；机组壳体内设置有自动控制系统和蒸发冷却‑机械制冷联合冷却系统，自动控制系统与蒸发冷却‑机械制冷联合冷却系统连接。本发明带自动控制系统的蒸发冷却‑机械制冷联合空调机组，利用自动控制系统实现了蒸发冷却与机械制冷联合空调机组内三种运行模式的准确切换，以便于适应不同的环境。'))
    patent_list = []
    num = 0
    with open('../data/patent_abstract/_bxk_abstract.txt', 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            line = line.strip()
            patent_list.append(line)
            num += 1
    # print(word_list)
    vector1 = model.infer_vector(word_list)
    # vector2 = np.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model.docvecs.vectors_docs.npy')
    print(vector1.shape)
    # print(vector2.shape)
    sims = model.docvecs.most_similar([vector1], topn=10)
    # sims = model.docvecs.most_similar([vector2[0]], topn=10)
    for i, sim in sims:
        print(i, sim)
        print(patent_list[i])
    # for sim in sims:
    #     print(sim[0])
    # print(vector1)