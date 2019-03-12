import re
import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import operator
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


class patent_ZH:
    def __init__(self, content, doc_num, ipc):
        self.label = -1
        self.content = content
        self.doc_num = doc_num
        self.docvec = None
        self.ipc = ipc

def get_label(file_list,cluster):
    f_num = 0
    for label in cluster:
        cur_file = file_list[f_num]
        cur_file.label = label
        f_num += 1
    return file_list

def get_patent_result(patent_list):
    result_dict = {}
    for patent in patent_list:
        if patent.label not in result_dict:
            result_dict[patent.label] = [patent.content]
        else:
            result_dict[patent.label].append(patent.content)
    result_dict = dict(sorted(result_dict.items(), key=operator.itemgetter(0)))
    return result_dict

def get_patent_ipc(patent_list):
    ipc_dict = {}
    for patent in patent_list:
        if patent.label not in ipc_dict:
            ipc_dict[patent.label] = [patent.ipc]
        else:
            ipc_dict[patent.label].append(patent.ipc)
    ipc_dict = dict(sorted(ipc_dict.items(), key=operator.itemgetter(0)))
    return ipc_dict

def get_class_num(labels):
    class_num = {}
    for label in labels:
        if label not in class_num:
            class_num[label] = 1
        else:
            class_num[label] += 1
    class_num = dict(sorted(class_num.items(), key=operator.itemgetter(0)))
    return class_num

if __name__ == '__main__':
    dim = 100
    model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10_5.model')
    patent_list = []
    docvecs = np.zeros((1, dim))
    num = 0
    # with open('D:\PycharmProjects\Dataset\keywordEX\patent\_all_label_abstract.txt', 'r', encoding='utf-8') as curf:
    #     for line in curf.readlines():
    #         content = re.sub('[，。；、]+', '', line)
    #         content = content.strip()
    #         each_cut = list(jieba.cut(content))
    #         line = line.strip()
    #         cur_patent = patent_ZH(line, num)
    #         cur_docvec = model.infer_vector(each_cut)
    #         cur_patent.docvec = cur_docvec
    #         print('读取第%d个专利摘要......' % (num + 1))
    #         if num == 0:
    #             docvecs[0] = cur_docvec.reshape(1, dim)
    #         else:
    #             docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, dim)))
    #         patent_list.append(cur_patent)
    #         num += 1
    with open('D:\PycharmProjects\Dataset\keywordEX\patent\_all_label_abstract.txt', 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            line_split = line.split(' ::  ')
            if len(line_split) == 2:
                content_rm = re.sub('[，。；、]+', '', line_split[1])
                content_rm = content_rm.strip()
                each_cut = list(jieba.cut(content_rm))
                content = line[1].strip()
                cur_patent = patent_ZH(content, num, line_split[0])
                cur_docvec = model.infer_vector(each_cut)
                cur_patent.docvec = cur_docvec
                print('读取第%d个专利摘要......' % (num + 1))
                if num == 0:
                    docvecs[0] = cur_docvec.reshape(1, dim)
                else:
                    docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, dim)))
                patent_list.append(cur_patent)
                num += 1
                if num >= 1000:
                    break
    print(docvecs.shape)
    # 1. 层次聚类
    # 生成点与点之间的距离矩阵,这里用的欧氏距离:
    disMat = sch.distance.pdist(docvecs, 'cosine')
    Z = sch.linkage(disMat, method='complete')
    # 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
    # plt.figure(num='层次聚类结果', figsize=(12, 8))
    # P=sch.dendrogram(Z)
    # plt.savefig('../data/patent_abstract/cengci/bxk_all_complete_100_10_5.png')
    # plt.savefig('../data/patent_abstract/cengci/Test.png')
    # 根据linkage matrix Z得到聚类结果:
    cluster = sch.fcluster(Z, 1.1, 'distance', depth=2)
    # ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
    # cluster = ac.fit_predict(docvecs)
    patent_list = get_label(patent_list, cluster)
    # my_result = get_patent_result(patent_list)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    # with open('../data/patent_abstract/cengci/bxk_all_100_10_5_cengci.txt', 'w', encoding='utf-8') as result_f:
    with open('../data/patent_abstract/cengci/Test.txt', 'w', encoding='utf-8') as result_f:
        result_f.write('聚类结果为：\n')
        for label in class_num:
            result_f.write(str(label) + ':' + str(class_num[label]) + '\n')
        for label in my_ipc:
            result_f.write('类标签为：' + str(label) + ':' + '\n')
            result_f.write(str(class_num[label]) + '条专利' + '\n')
            for ipc in my_ipc[label]:
                result_f.write(ipc + ' ;' + '\n')
        # for label in my_result:
        #     result_f.write('类标签为：' + str(label) + ':' + '\n')
        #     result_f.write(str(class_num[label]) + '条专利' + '\n')
        #     for patent in my_result[label]:
        #         result_f.write(patent + ' ;' + '\n')

