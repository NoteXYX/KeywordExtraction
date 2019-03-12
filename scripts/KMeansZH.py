import re
import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import operator
from sklearn.cluster import KMeans


class patent_ZH:
    def __init__(self, content, doc_num):
        self.label = -1
        self.doc_num = doc_num
        self.docvec = None
        self.content = content

def get_label(file_list,cluster):
    f_num = 0
    for label in cluster:
        cur_file = file_list[f_num]
        cur_file.label = label
        f_num += 1
    return file_list

def get_patent_result(patent_list):
    my_dict = {}
    for patent in patent_list:
        if patent.label not in my_dict:
            my_dict[patent.label] = [patent.content]
        else:
            my_dict[patent.label].append(patent.content)
    my_dict = dict(sorted(my_dict.items(), key=operator.itemgetter(0)))
    return my_dict

def get_class_num(labels):
    class_num = {}
    for label in labels:
        if label not in class_num:
            class_num[label] = 1
        else:
            class_num[label] += 1
    class_num = dict(sorted(class_num.items(), key=operator.itemgetter(0)))
    return class_num

def get_class_title(labels):
    class_title = {}
    for i, label in enumerate(labels):
        if label not in class_title:
            class_title[label] = [i]
        else:
            class_title[label].append(i)
    class_title = dict(sorted(class_title.items(), key=operator.itemgetter(0)))
    return class_title

if __name__ == '__main__':
    dim = 200
    model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_200_dm_10_5.model')
    patent_list = []
    docvecs = np.zeros((1,dim))
    num = 0
    with open('../data/patent_abstract/_bxk_abstract.txt', 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            content = re.sub('[，。；、]+', '', line)
            content = content.strip()
            each_cut = list(jieba.cut(content))
            line = line.strip()
            cur_patent = patent_ZH(line, num)
            cur_docvec = model.infer_vector(each_cut)
            cur_patent.docvec = cur_docvec
            print('读取第%d个专利摘要......' % (num + 1))
            if num == 0:
                docvecs[0] = cur_docvec.reshape(1,dim)
            else:
                docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, dim)))
            patent_list.append(cur_patent)
            num += 1
    print(docvecs.shape)
    cluster = KMeans(n_clusters=3, random_state=9, max_iter=300).fit_predict(docvecs)
    patent_list = get_label(patent_list, cluster)
    my_result = get_patent_result(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    # class_title = get_class_title(cluster)
    with open('../data/patent_abstract/bxk_all_200_dm_10_5_KMeans.txt', 'w', encoding='utf-8') as result_f:
        result_f.write('聚类结果为：\n')
        for label in class_num:
            result_f.write(str(label) + ':' + str(class_num[label]) + '\n')
        for label in my_result:
            result_f.write('类标签为：' + str(label) + ':' +'\n')
            result_f.write(str(class_num[label]) + '条专利' + '\n')
            for patent in my_result[label]:
                result_f.write(patent + ' ;' + '\n')


