import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets
from embeddings import read
from sklearn.manifold import TSNE


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
    embedding_file = open(r'..\data\model\word2vec\patent\bxk_200_SG.vector', 'r', encoding='utf-8', errors='surrogateescape')
    words, vectors = read(embedding_file, dtype=float)
    plot_only = 5000
    log_file = open('../data/patent_log.txt', 'a', encoding='utf-8')
    myeps = 0.01
    while myeps <= 2.0:
        for my_min_samples in range(3, 11):
            db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples).fit_predict(vectors)
            class_num = get_class_num(db_labels)
            print('eps=%f, min_samples=%d' % (myeps, my_min_samples))
            n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            print('聚类的类别数目(除噪音外)：%d' % (n_clusters_))
            ratio = len(db_labels[db_labels[:] == -1]) / len(db_labels)
            print('噪音率:' + str(ratio))
            log_file.write('eps = %f ,min_samples = %d \n聚类的类别数目（除噪音外）：%d , 噪音率: %f\n' % (myeps, my_min_samples, n_clusters_, ratio))
            print('聚类结果为：')
            log_file.write('聚类结果为：\n')
            for label in class_num:
                print(str(label) + ':' + str(class_num[label]))
                log_file.write(str(label) + ':' + str(class_num[label]) + '\t;\t')
            print('----------------------------------------------------------------')
            log_file.write('\n------------------------------------------------------------------\n')
        myeps = myeps + 0.01


    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    # low_dim_embs = tsne.fit_transform(vectors[:plot_only, :])
    # print(low_dim_embs.shape)
    # labels = [words[i] for i in range(plot_only)]
    # plot_with_labels(low_dim_embs, colors, labels, '../data/DBSCAN_SE2010.png')

    embedding_file.close()
    log_file.close()
