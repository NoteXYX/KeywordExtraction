import logging
import os
import sys
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets
from embeddings import read
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.manifold import TSNE
from sklearn.cluster import Birch

def plot_with_labels(low_dim_embs, colors, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(50, 50))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, c=colors[i])
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

def get_class_num(labels):
    class_num = {}
    for label in labels:
        if label not in class_num:
            class_num[label] = 1
        else:
            class_num[label] += 1
    class_num = dict(sorted(class_num.items(), key=operator.itemgetter(0)))
    return class_num
# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                       noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
# X = np.concatenate((X1, X2))
# y_pred = [-1 for i in range(6000)]
# plt.scatter(X[:, 0], X[:, 1], marker='o',c=y_pred)
# plt.show()
# y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
# y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
# print(y_pred.shape)
# n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
# print('聚类的类别数目：%d' % (n_clusters_))
# ratio = len(y_pred[y_pred[:] == -1]) / len(y_pred)
# print('认为是噪音的数据比例：%d' % (ratio))
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()


if __name__ == '__main__':
    # embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\old\all_50_SG.vector', 'r', encoding='utf-8', errors='surrogateescape')
    # model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model')
    # embedding_file = open(r'D:\PycharmProjects\KeywordExtraction\data\model\word2vec\patent\bxk_50_SG.vector', 'r', encoding='utf-8', errors='surrogateescape')
    # words, vectors = read(embedding_file, dtype=float)
    # plot_only = 5000
    # log_file = open('../data/allpatent_log.txt', 'a', encoding='utf-8')
    # myeps = 2
    # while myeps <= 2.5:
    #     for my_min_samples in range(5,7):
    #         print('DBSCAN聚类中......')
    #         db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples, n_jobs=-1 ).fit_predict(vectors)
    #         # db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples, algorithm='ball_tree').fit_predict(vectors)
    #         class_num = get_class_num(db_labels)
    #         print('eps=%f, min_samples=%d' % (myeps, my_min_samples))
    #         n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    #         print('聚类的类别数目(除噪音外)：%d' % (n_clusters_))
    #         ratio = len(db_labels[db_labels[:] == -1]) / len(db_labels)
    #         print('噪音率:' + str(ratio))
    #         log_file.write('eps = %f ,min_samples = %d \n聚类的类别数目（除噪音外）：%d , 噪音率: %f\n' % (myeps, my_min_samples, n_clusters_, ratio))
    #         print('聚类结果为：')
    #         log_file.write('聚类结果为：\n')
    #         for label in class_num:
    #             print(str(label) + ':' + str(class_num[label]))
    #             log_file.write(str(label) + ':' + str(class_num[label]) + '\t;\t')
    #         print('----------------------------------------------------------------')
    #         log_file.write('\n------------------------------------------------------------------\n')
    #     myeps = myeps + 0.1

    # # 1. 层次聚类
    # # 生成点与点之间的距离矩阵,这里用的欧氏距离:
    # disMat = sch.distance.pdist(vectors, 'cosine')
    # Z = sch.linkage(disMat, method='average')
    # # 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
    # P=sch.dendrogram(Z)
    # plt.savefig('plot_dendrogram.png')
    # # 根据linkage matrix Z得到聚类结果:
    # cluster = sch.fcluster(Z, 0.5, 'distance', depth=2)
    # P=sch.dendrogram(Z)
    # plt.savefig('plot_dendrogram.png')
    # ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
    # cluster = ac.fit_predict(vectors)
    # labels_unique = np.unique(cluster)
    # n_clusters_ = len(labels_unique)
    # print('聚类的类别数目：%d' % n_clusters_)

    # #2.MeanShift
    # bandwidth = estimate_bandwidth(vectors, quantile=10, n_samples=10)
    # ##设置均值偏移函数
    # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ##训练数据
    # ms.fit(vectors)
    # ##每个点的标签
    # labels = ms.labels_
    # ##总共的标签分类
    # labels_unique = np.unique(labels)
    # ##聚簇的个数，即分类的个数
    # n_clusters_ = len(labels_unique)
    # print('聚类的类别数目：%d' % n_clusters_)
    # print('聚类结果为：')
    # class_num = get_class_num(labels)
    # for label in class_num:
    #     print(str(label) + ':' + str(class_num[label]))
    # print('----------------------------------------------------------------')

    # #3.Kmeans
    # for n_clusters_ in range(3,11):
    #     print('Kmeans聚类中......')
    #     db_labels = KMeans(n_clusters=n_clusters_, random_state=9).fit_predict(vectors)
    #     class_num = get_class_num(db_labels)
    #     print('聚类的类别数目=%d' % n_clusters_)
    #     print('聚类结果为：')
    #     for label in class_num:
    #         print(str(label) + ':' + str(class_num[label]))
    #     print('----------------------------------------------------------------')
    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    # low_dim_embs = tsne.fit_transform(vectors[:plot_only, :])
    # print(low_dim_embs.shape)
    # labels = [words[i] for i in range(plot_only)]
    # plot_with_labels(low_dim_embs, colors, labels, '../data/DBSCAN_SE2010.png')

    # # 4.BIRCH
    # my_threshold = 0.5
    # while my_threshold <= 1.0:
    #     for my_branching in range(50,80,10):
    #         print('BIRCH聚类中......')
    #         birch_labels = Birch(n_clusters = None, threshold = my_threshold, branching_factor = my_branching).fit_predict(vectors)
    #         class_num = get_class_num(birch_labels)
    #         labels_unique = np.unique(birch_labels)
    #         n_clusters_ = len(labels_unique)
    #         print('聚类的类别数目=%d' % n_clusters_)
    #         print('聚类结果为：')
    #         for label in class_num:
    #             print(str(label) + ':' + str(class_num[label]))
    #         print('----------------------------------------------------------------')
    # birch_labels = Birch(n_clusters=4, threshold=0.5, branching_factor=50).fit_predict(vectors)
    # class_num = get_class_num(birch_labels)
    # labels_unique = np.unique(birch_labels)
    # n_clusters_ = len(labels_unique)
    # print('聚类的类别数目=%d' % n_clusters_)
    # print('聚类结果为：')
    # for label in class_num:
    #     print(str(label) + ':' + str(class_num[label]))
    # embedding_file.close()

    # # 5.Doc2vec
    # sentvecs = np.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model.docvecs.vectors_docs.npy')
    # print(sentvecs.shape)
    # log_file = open('../data/all_Doc2vec_log.txt', 'a', encoding='utf-8')
    # myeps = 2
    # while myeps <= 2.5:
    #     for my_min_samples in range(5,7):
    #         print('DBSCAN聚类中......')
    #         # db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples, n_jobs=-1 ).fit_predict(sentvecs)
    #         db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples, algorithm='ball_tree').fit_predict(sentvecs)
    #         class_num = get_class_num(db_labels)
    #         print('eps=%f, min_samples=%d' % (myeps, my_min_samples))
    #         n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    #         print('聚类的类别数目(除噪音外)：%d' % (n_clusters_))
    #         ratio = len(db_labels[db_labels[:] == -1]) / len(db_labels)
    #         print('噪音率:' + str(ratio))
    #         log_file.write('eps = %f ,min_samples = %d \n聚类的类别数目（除噪音外）：%d , 噪音率: %f\n' % (myeps, my_min_samples, n_clusters_, ratio))
    #         print('聚类结果为：')
    #         log_file.write('聚类结果为：\n')
    #         for label in class_num:
    #             print(str(label) + ':' + str(class_num[label]))
    #             log_file.write(str(label) + ':' + str(class_num[label]) + '\t;\t')
    #         print('----------------------------------------------------------------')
    #         log_file.write('\n------------------------------------------------------------------\n')
    #     myeps = myeps + 0.1
    # log_file.close()
    sentvecs = np.load(r'D:\PycharmProjects\KeywordExtraction\data\model\sen2vec\patent\bxkdoc_100_dm_20.vector.npy')
    print(sentvecs.shape)
    myeps = 2
    while myeps <= 2.5:
        for my_min_samples in range(3,6):
            print('DBSCAN聚类中......')
            # db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples, n_jobs=-1 ).fit_predict(sentvecs)
            db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples).fit_predict(sentvecs)
            class_num = get_class_num(db_labels)
            print('eps=%f, min_samples=%d' % (myeps, my_min_samples))
            n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            print('聚类的类别数目(除噪音外)：%d' % (n_clusters_))
            ratio = len(db_labels[db_labels[:] == -1]) / len(db_labels)
            print('噪音率:' + str(ratio))
            print('聚类结果为：')
            for label in class_num:
                print(str(label) + ':' + str(class_num[label]))
            print('----------------------------------------------------------------')
        myeps = myeps + 0.1
