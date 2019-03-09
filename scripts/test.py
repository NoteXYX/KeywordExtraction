import numpy as np
import re
import os
import jieba
from gensim.models.doc2vec import Doc2Vec

if __name__ == '__main__':
    model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model')
    word_list = list(jieba.cut('该技术方案能够完全摆脱遥控器实现对空调的控制，操作方便，同时，语音交互方式具有灵活性，能够满足不同用户个性化的要求，提高了用户的体验'))
    print(word_list)
    vector1 = model.infer_vector(word_list)
    vector2 = np.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model.docvecs.vectors_docs.npy')
    print(vector1.shape)
    print(vector2.shape)
    sims = model.docvecs.most_similar([vector1], topn=10)
    # sims = model.docvecs.most_similar([vector2[0]], topn=10)
    for i, sim in sims:
        print(i, sim)
    # for sim in sims:
    #     print(sim[0])
    print(vector1)