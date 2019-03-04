import numpy as np
import operator
from sklearn.datasets.samples_generator import make_blobs
from gensim.models.doc2vec import Doc2Vec
import jieba

print(1.5<=1.5)

# model = Doc2Vec.load('../data/model/sen2vec/patent/bxk_50_dm_40.model')
#
# word_list = list(jieba.cut('该技术方案能够完全摆脱遥控器实现对空调的控制，操作方便，同时，语音交互方式具有灵活性，能够满足不同用户个性化的要求，提高了用户的体验'))
# print(word_list)
# vector1 = model.infer_vector(word_list)
# vector2 = np.load('../data/model/sen2vec/patent/bxk_50_dm_40.npy')
# print(vector1.shape)
# print(vector2.shape)
# sims = model.docvecs.most_similar([vector1], topn=10)
# for i, sim in sims:
#     print(i, sim)
# # for sim in sims:
# #     print(sim[0])
# print(vector1)
#
# print(vector2[0])
# num = float(np.dot(vector2[0], vector1.reshape(1, 100).T))
# vec_norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
# cos = num / vec_norm
# sim = 0.5 + 0.5 * cos   # 归一化
# print(sim)