import numpy as np
import operator
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn import metrics
from embeddings import read, plot_with_labels
from sklearn.manifold import TSNE


def get_DBSCAN_clusters(vectors,labels):    # 根据DBSCAN聚类后的标签labels整理各类的向量，存放在字典clusters
    clusters = {}
    for i in range(len(labels)):
        if labels[i] not in clusters:
            clusters[labels[i]] = vectors[i]
        elif labels[i] in clusters:
            cur_vec = vectors[i]
            cur_cluster = clusters[labels[i]]
            clusters[labels[i]] = np.row_stack((cur_cluster, cur_vec))
    return clusters

def get_centers(model, clusters, method):  # 获得各个类的中心点(噪音类除外)
    centers = {}
    if method == 'DBSCAN':
        for label in clusters:
            if label == -1:     #如果是噪音类
                continue
            else:
                cur_vectors = clusters[label]
                cur_center = np.mean(cur_vectors, 0)
                centers[label] = cur_center
    elif method == 'Kmeans':
        label = 0
        for center in model.cluster_centers_:
            centers[label] = center
            label += 1
    return centers


def get_distance(cur_vector, cur_center, method):   # 获得与中心点的距离(余弦相似度 or 欧式距离)
    if method == 'cos':
        num = float(np.dot(cur_vector, cur_center.T))
        vec_norm = np.linalg.norm(cur_vector) * np.linalg.norm(cur_center)
        cos = num / vec_norm
        sim = 0.5 + 0.5 * cos   # 归一化
        return sim
    elif method == 'ED':
        distance = np.linalg.norm(cur_vector - cur_center)
        return distance

def distance_sort(ind2vec, cur_center, method):     # 获得根据与中心点距离大小排序后的{词向量：与中心点的距离}
    index_distance = {}
    for index in ind2vec:
        distance = get_distance(ind2vec[index], cur_center, method)
        index_distance[index] = distance
    if method == 'cos':
        sorted_distance = sorted(index_distance.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_distance = sorted(index_distance.items(), key=operator.itemgetter(1))
    sorted_index_distance = dict(sorted_distance)
    return sorted_index_distance

def get_index2vectors(filename, word2ind, wordvecs):    # 获得测试文本中所有词的词向量
    ind2vec = {}
    test_file = open(filename, 'r', encoding='utf-8')
    for line in test_file.readlines():
        curline_words = line.split(' ')
        for word in curline_words:
            if word == '\n':
                continue
            elif word in word2ind:
                cur_index = word2ind[word]
                cur_vec = wordvecs[cur_index]
                ind2vec[cur_index] = cur_vec
    test_file.close()
    return ind2vec

def get_most_label(ind2vec, clusters):     # 获得测试文本中单词数最多的类别
    class_vector = {}
    for index in ind2vec:
        for label in clusters:
            if ind2vec[index] in clusters[label]:
                if label not in class_vector:
                    class_vector[label] = ind2vec[index]
                else:
                    class_vector[label] = np.row_stack((class_vector[label], ind2vec[index]))
                break
    assert len(class_vector) > 0
    class_vector = dict(sorted(class_vector.items(), key=operator.itemgetter(0)))
    if len(class_vector) == 1:
        most_label = -1
        print('所有词向量均为噪音！')
        return most_label
    else:
        most_label = 0
        most_num = class_vector[most_label].shape[0]
    for label in class_vector:
        if label == -1 or label == 0:
            continue
        else:
            if class_vector[label].shape[0] > most_num:
                most_num = class_vector[label].shape[0]
                most_label = label
    print('本文中%d类包含的单词最多，单词数为：%d,占本文单词的%f%%' % (most_label, most_num, most_num * 100.0 / len(ind2vec)))
    return most_label

def read_corpus(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read().split('.')
    return content
def main():
    model = Doc2Vec.load('../data/model/sen2vec/SE2010_200.model')
    content = read_corpus('../data/SE2010_content.txt')
    vectors = np.load('../data/model/sen2vec/SE2010_200.model.docvecs.vectors_docs.npy')
    db_model = DBSCAN(eps=1.98, min_samples=3).fit(vectors)
    # vector = model.infer_vector('a challenging problem faced by researchers and developers'.split(' '))
    # sims = model.docvecs.most_similar([vector], topn=20)
    sims = model.docvecs.most_similar([vectors[0]], topn=20)
    for count, sim in sims:
        sentence = content[count]
        print(count)
        print(sentence)
        print(sim)
        print('--------------------------------------------------------')





if __name__ == '__main__':
    main()

