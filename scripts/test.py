import numpy as np
import re
import os
import jieba
from gensim.models.doc2vec import Doc2Vec

if __name__ == '__main__':
    model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model')
    word_list = list(jieba.cut('本发明涉及一种由甲硝唑制备的呋喃酮苯胺类衍生物及其制法与在抗菌药物中的应用。化合物名称为4‑((3，5‑二甲氧基苯基)氨基)‑3‑(2‑甲基‑5‑硝基‑1H‑咪唑‑1‑基)呋喃‑2(5H)‑酮。本发明所述的化合物对所测试的细菌均表现出较好的抑制和杀灭作用，对枯草芽孢杆菌的抑制活性接近阳性对照卡那霉素，对表面葡萄球菌的抑制活性超过了阳性对照卡那霉素，因此可用于制备抗感染药物；本发明利用呋喃酮环代替先导化合物的丙烯酸酯部分改良了构型互变带来的影响，并引入了具有较强抗菌作用的类甲硝唑结构，在深入研究构效关系的基础上，设计合成了活性更高的新型抗菌化合物，并提供了所述化合物的制备方法。'))
    patent_list = []
    num = 0
    with open('D:\PycharmProjects\Dataset\keywordEX\patent\_all_abstract.txt', 'r', encoding='utf-8') as curf:
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