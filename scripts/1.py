import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn import metrics
from embeddings import read, plot_with_labels
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
import scipy
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from sklearn.datasets.samples_generator import make_blobs
# X1, y1 = datasets.make_blobs(n_samples=100, n_features=2, centers=[[0.5,0.5]], cluster_std=[[.1]],
#                random_state=9)
# X2, y2 = datasets.make_blobs(n_samples=100, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                           noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
# # points = np.concatenate((X1, X2))
# points=scipy.randn(10000,50)
#1. 层次聚类
#生成点与点之间的距离矩阵,这里用的欧氏距离:
# disMat = sch.distance.pdist(points,'euclidean')
#进行层次聚类:
# Z=sch.linkage(points,method='average', metric='euclidean')
# #将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
#
# #根据linkage matrix Z得到聚类结果:
# cluster= sch.fcluster(Z, 0.4, 'distance', depth=2)
# # P=sch.dendrogram(Z)
# # plt.savefig('plot_dendrogram.png')
# model=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='average')
# cluster = model.fit_predict(points)
# n_clusters_ = len(set(cluster))
# print('聚类的类别/数目：%d' % (n_clusters_))
# print("Original cluster by hierarchy clustering:\n",cluster)
# plt.scatter(points[:, 0], points[:, 1], c=cluster)
# plt.show()

# #2. MeanShift
# ##带宽，也就是以某个点为核心时的搜索半径
# bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=500)
# ##设置均值偏移函数
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ##训练数据
# ms.fit(points)
# ##每个点的标签
# labels = ms.labels_
# print(labels)
# ##簇中心的点的集合
# cluster_centers = ms.cluster_centers_
# ##总共的标签分类
# labels_unique = np.unique(labels)
# ##聚簇的个数，即分类的个数
# n_clusters_ = len(labels_unique)
# plt.scatter(points[:, 0], points[:, 1], c=labels)
# plt.show()

# #3. 层次聚类
# ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
# cluster = ac.fit_predict(points)
# labels_unique = np.unique(cluster)
# n_clusters_ = len(labels_unique)
# print('聚类的类别数目：%d' % n_clusters_)
# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                           noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
# X = np.concatenate((X1, X2))
# y_pred = [-1 for i in range(6000)]
# # plt.scatter(X[:, 0], X[:, 1], marker='o',c=y_pred)
# # plt.show()
# # y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
# y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
# print(y_pred.shape)
# n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
# print('聚类的类别数目：%d' % (n_clusters_))
# ratio = len(y_pred[y_pred[:] == -1]) / len(y_pred)
# print('认为是噪音的数据比例：%d' % (ratio))
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()

# # 4.BIRCH
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3],
                  random_state =9)
from sklearn.cluster import Birch
y_pred = Birch(n_clusters = None, threshold = 0.5, branching_factor = 60).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
from sklearn import metrics
print ("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))

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