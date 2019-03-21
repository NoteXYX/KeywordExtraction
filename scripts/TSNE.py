import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import jieba
from embeddings import read
from sklearn.cluster import Birch
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',ha='right', va='bottom')
    plt.savefig(filename)
#def read(file, threshold=0, vocabulary=None, dtype='float'):
#    header = file.readline().split(' ')
#    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
#    dim = int(header[1])
#    words = []
#    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
#    for i in range(count):
#        word, vec = file.readline().split(' ', 1)
#        if vocabulary is None:
#            words.append(word)
#            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
#        elif word in vocabulary:
#            words.append(word)
#            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
#    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))

if __name__ == '__main__':
    try:
        # pylint: disable=g-import-not-at-top
        #embedding_file = open('../data/model/SE2010.vector', encoding='utf-8', errors='surrogateescape')
        #words, vectors = read(embedding_file, dtype=float)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_techField_NEW.vec', 'r',
                              encoding='utf-8', errors='surrogateescape')
        stop_file = open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8')
        stopwords = list()
        dim = 100
        for line in stop_file.readlines():
            stopwords.append(line.strip())
        words, wordvecs = read(embedding_file, dtype=float)
        word2ind = {word: i for i, word in enumerate(words)}
        bxdwords = list()
        tsne_vecs = np.zeros((1, dim))
        with open('D:\PycharmProjects\Dataset\keywordEX\patent\_bxd_label_techField.txt', 'r', encoding='utf-8') as test_file:
            num = 0
            for test_line in test_file.readlines():
                line_split = test_line.split(' ::  ')
                if len(line_split) == 2:
                    content = line_split[1].strip()
                    test_line_words = list(jieba.cut(content))
                    line_words = [word for word in test_line_words if word not in stopwords]
                    line_wordvecs = np.zeros((1, dim))
                    for i in range(len(line_words)):
                        if line_words[i] in word2ind and line_words[i] not in bxdwords:
                            cur_wordindex = word2ind[line_words[i]]
                            cur_wordvec = wordvecs[cur_wordindex].reshape(1, dim)
                            if num == 0:
                                tsne_vecs[0] = cur_wordvec
                            else:
                                tsne_vecs = np.row_stack((tsne_vecs, cur_wordvec))
                            bxdwords.append(line_words[i])
                            num += 1
                            print('处理第%d个词向量......' % num)
        print(tsne_vecs.shape)
        # cluster = Birch(n_clusters=3, threshold=0.7, branching_factor=50).fit_predict(tsne_vecs)
        #plot_only = 1000
        low_dim_embs = tsne.fit_transform(tsne_vecs)
        #labels = [words[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, bxdwords, '../data/TSNE.png')

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)
