import numpy as np
import operator
from sklearn.datasets.samples_generator import make_blobs
from gensim.models.doc2vec import Doc2Vec
import jieba


model = Doc2Vec.load('../data/model/sen2vec/patent/bxk_50_dm_20.model')

word_list = list(jieba.cut('本发明公开一种具有语音交互功能的声控空调器，通过用户发出的语音指令信息直接对空调器进行控制，并在对空调进行语音控制过程中通过反馈语音指令信息给用户确认，实现用户与空调的语音交互'))
print(word_list)
vector1 = model.infer_vector(word_list)
vector2 = np.load('../data/model/sen2vec/patent/bxk_50_dm_20.npy')
sims = model.docvecs.most_similar([vector1], topn=10)
for i, sim in sims:
    print(i, sim)
print(vector1)

print(vector2[0])