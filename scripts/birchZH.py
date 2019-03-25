import re
import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import operator
from embeddings import read
from sklearn.cluster import Birch
from sklearn import metrics


class patent_ZH:
    def __init__(self, content, doc_num, ipc):
        self.label = -1
        self.content = content
        self.doc_num = doc_num
        self.docvec = None
        self.ipc = ipc

def get_label(patent_list,cluster):
    f_num = 0
    for label in cluster:
        cur_file = patent_list[f_num]
        cur_file.label = label
        f_num += 1
    return patent_list

def get_patent_result(patent_list):
    result_dict = dict()
    for patent in patent_list:
        if patent.label not in result_dict:
            result_dict[patent.label] = [patent.content]
        else:
            result_dict[patent.label].append(patent.content)
    result_dict = dict(sorted(result_dict.items(), key=operator.itemgetter(0)))
    return result_dict

def get_patent_ipc(patent_list):
    ipc_dict = dict()
    for patent in patent_list:
        if patent.label not in ipc_dict:
            ipc_dict[patent.label] = [patent.ipc]
        else:
            ipc_dict[patent.label].append(patent.ipc)
    ipc_dict = dict(sorted(ipc_dict.items(), key=operator.itemgetter(0)))
    return ipc_dict

def get_class_num(labels):
    class_num = dict()
    for label in labels:
        if label not in class_num:
            class_num[label] = 1
        else:
            class_num[label] += 1
    class_num = dict(sorted(class_num.items(), key=operator.itemgetter(0)))
    return class_num

def get_clusters(vectors,labels):    # 根据DBSCAN聚类后的标签labels整理各类的向量，存放在字典clusters
    clusters = {}
    for i in range(len(labels)):
        if labels[i] not in clusters:
            clusters[labels[i]] = vectors[i]
        elif labels[i] in clusters:
            cur_vec = vectors[i]
            cur_cluster = clusters[labels[i]]
            clusters[labels[i]] = np.row_stack((cur_cluster, cur_vec))
    return clusters

def get_centers(clusters):  # 获得各个类的中心点(噪音类除外)
    centers = {}
    for label in clusters:
        if label == -1:     #如果是噪音类
            continue
        else:
            cur_vectors = clusters[label]
            km_model = KMeans(n_clusters=1, max_iter=500, random_state=0).fit(cur_vectors)
            km_labels = km_model.labels_
            km_score = metrics.calinski_harabaz_score(cur_vectors, km_labels)
            print('类标签为%d的K-means聚类得分：%f' % (label, km_score))
            cur_center = km_model.cluster_centers_
            print('类标签为%d的K-means聚类中心：' %label + str(cur_center))
            centers[label] = cur_center
    return centers

def get_distance(cur_vector, cur_center, method):   # 获得与中心点的距离(余弦相似度 or 欧式距离)
    if method == 'cos':
        num = float(np.dot(cur_vector, cur_center.T))
        vec_norm = np.linalg.norm(cur_vector) * np.linalg.norm(cur_center)
        cos = num / vec_norm
        sim = 0.5 + 0.5 * cos   # 归一化
        return sim
    elif method == 'ED':
        dist = np.linalg.norm(cur_vector - cur_center)
        return dist

def distance_sort(vectors, cur_center, method):     # 获得根据与中心点距离大小排序后的{词向量：与中心点的距离}
    distance_dict = {}
    for vector in vectors:
        distance = get_distance(vector, cur_center, method)
        distance_dict[vector] = distance
    sorted_distance = sorted(distance_dict.items(), key=operator.itemgetter(1))
    sorted_distance_dict = dict(sorted_distance)
    return sorted_distance_dict

def get_stopwords():
    stop_file = open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8')
    stopwords = list()
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    return stopwords

def write_cluster_result(fname, class_num, my_ipc):
    with open(fname, 'w', encoding='utf-8') as result_f:
        result_f.write('聚类结果为：\n')
        for label in class_num:
            result_f.write(str(label) + ':' + str(class_num[label]) + '\n')
        for label in my_ipc:
            result_f.write('类标签为:' + str(label) + ':' + '\n')
            result_f.write(str(class_num[label]) + '条专利' + '\n')
            for ipc in my_ipc[label]:
                result_f.write(str(label) + ':  ' + ipc + '\n')

def get_most_label(line_vecs, birch_model):
    # word_cluster = dict()
    label_num = dict()
    for vec in line_vecs:
        cur_label = birch_model.predict(vec)
        # print(cur_label)
        if cur_label[0] not in label_num:
            label_num[cur_label[0]] = 1
        else:
            label_num[cur_label[0]] += 1
        # word_cluster[vec] = cur_label
    label_num = dict(sorted(label_num.items(), key=operator.itemgetter(1), reverse=True))
    most_label = list(label_num.items())[0][0]
    return most_label

def birch1():       # Doc2vec
    dim = 100
    model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_abstract_100_nostop.model')
    patent_list = list()
    docvecs = np.zeros((1, dim))
    num = 0
    stopwords = get_stopwords()
    with open('D:\PycharmProjects\Dataset\keywordEX\patent\_bxk_label_abstract.txt', 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            line_split = line.split(' ::  ')
            if len(line_split) == 2:
                content_rm = line_split[1].strip()
                line_cut = list(jieba.cut(content_rm))
                line_words = [word for word in line_cut if word not in stopwords]
                content = line_split[1].strip()
                cur_patent = patent_ZH(content, num, line_split[0])
                cur_docvec = model.infer_vector(line_words)
                cur_patent.docvec = cur_docvec
                print('读取第%d个专利摘要......' % (num + 1))
                if num == 0:
                    docvecs[0] = cur_docvec.reshape(1, dim)
                else:
                    docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, dim)))
                patent_list.append(cur_patent)
                num += 1
    print(docvecs.shape)
    model = Birch(n_clusters=3, threshold=0.5, branching_factor=50).fit(docvecs)
    cluster = model.labels_
    patent_list = get_label(patent_list, cluster)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    write_cluster_result('../data/patent_abstract/Brich/bxk_abstract_nostop_doc2vecTest_100.txt', class_num, my_ipc)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(docvecs, cluster))
    return model

def birch2():       # sent2vec
    embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\sent2vec\bxd_fc_rm_abstract.vec', 'r',
                          encoding='utf-8', errors='surrogateescape')
    sent_num, sentvecs = read(embedding_file, dtype=float)
    patent_list = list()
    num = 0
    dim = 100
    with open('D:\PycharmProjects\Dataset\keywordEX\patent\_bxd_label_abstract.txt', 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            line_split = line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                cur_patent = patent_ZH(content, num, line_split[0])
                cur_patent.docvec = sentvecs[num].reshape(1, dim)
                patent_list.append(cur_patent)
                print('读取第%d个专利摘要......' % (num + 1))
                num += 1
    print(sentvecs.shape)
    model = Birch(n_clusters=3, threshold=0.7, branching_factor=50).fit(sentvecs)
    cluster = model.labels_
    patent_list = get_label(patent_list, cluster)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    write_cluster_result('../data/patent_abstract/Brich/bxd_abstract_sent2vec_Test.txt', class_num, my_ipc)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(sentvecs, cluster))
    embedding_file.close()
    return model

def birch3():       # 词向量加和平均
    embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_techField_100.vec', 'r',
                          encoding='utf-8', errors='surrogateescape')
    patent_list = list()
    dim = 100
    stopwords = get_stopwords()
    words, wordvecs = read(embedding_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    test_vecs = np.zeros((1, dim))
    with open('D:\PycharmProjects\Dataset\keywordEX\patent\_bxd_label_techField.txt', 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                cur_patent = patent_ZH(content, num, line_split[0])
                test_line_words = list(jieba.cut(content))
                line_words = [word for word in test_line_words if word not in stopwords]
                line_wordvecs = np.zeros((1, dim))
                for i in range(len(line_words)):
                    if line_words[i] in word2ind:
                        cur_wordindex = word2ind[line_words[i]]
                        cur_wordvec = wordvecs[cur_wordindex].reshape(1, dim)
                        if i == 0:
                            line_wordvecs[0] = cur_wordvec
                        else:
                            line_wordvecs = np.row_stack((line_wordvecs, cur_wordvec))
                cur_linevec = np.mean(line_wordvecs, axis=0).reshape(1, dim)
                cur_patent.docvec = cur_linevec
                patent_list.append(cur_patent)
                test_vecs = np.row_stack((test_vecs, cur_linevec))
                print('处理第%d条专利......' % (num+1))
            num += 1
        test_vecs = np.delete(test_vecs, 0 , 0)
    print(test_vecs.shape)
    model = Birch(n_clusters=3, threshold=0.7, branching_factor=50).fit(test_vecs)
    cluster = model.labels_
    patent_list = get_label(patent_list, cluster)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    write_cluster_result('../data/patent_abstract/Brich/keyword_test.txt', class_num, my_ipc)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(test_vecs, cluster))
    embedding_file.close()
    return model

def keyword_extraction(test_name, wordvec_file, birch_model, dim):
    stopwords = get_stopwords()
    words, wordvecs = read(wordvec_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    with open(test_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                test_line_words = list(jieba.cut(content))
                line_words = list()
                line_vecs = list()
                for word in test_line_words:
                    if word not in stopwords and word in word2ind:
                        line_words.append(word)
                        cur_wordvec = wordvecs[word2ind[word]].reshape(1, dim)
                        line_vecs.append(cur_wordvec)
                assert len(line_words) == len(line_vecs)
                get_most_label(line_vecs, birch_model)
                most_label = get_most_label(line_vecs, birch_model)
                print(most_label)
                num += 1
                if num >= 10:
                    break


if __name__ == '__main__':
    dim = 100
    wordvec_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_techField_100.vec', 'r', encoding='utf-8', errors='surrogateescape')
    test_name = 'D:\PycharmProjects\Dataset\keywordEX\patent\_bxd_label_techField.txt'
    # birch1()
    # birch2()
    birch_model = birch3()
    keyword_extraction(test_name, wordvec_file, birch_model, dim)
    wordvec_file.close()



