import numpy as np
import re
import os
import jieba
from gensim.models.doc2vec import Doc2Vec

if __name__ == '__main__':
    my_ipc = dict()
    ipc_num = 0
    with open('../data/patent_abstract/Birch/bxd_techField_wordAVG_keywordTest_1.04_50.txt', 'r', encoding='utf-8') as result_f:
    # with open('../data/patent_abstract/Kmeans/bxk_techField_word2vecAVG_Test.txt', 'r', encoding='utf-8') as result_f:
    # with open('../data/patent_abstract/cengci/bxd_abstract_nostop_doc2vecTest_100.txt', 'r', encoding='utf-8') as result_f:
        result_lines = result_f.readlines()
        line_num = 0
        if_write = False
        cur_label = -1
        while line_num < len(result_lines):
            search_title = re.search('类标签为:', result_lines[line_num])
            if search_title:
                cur_label = int(result_lines[line_num].split(':')[1])
                if_write = True
                line_num += 2
            if if_write:
                if cur_label not in my_ipc:
                    my_ipc[cur_label] = [result_lines[line_num]]
                    ipc_num += 1
                else:
                    my_ipc[cur_label].append(result_lines[line_num])
                    ipc_num += 1
                line_num += 1
            else:
                line_num += 1
    # truth = {0: 'H04N', 1: 'F24F', 2: 'B08B'}
    # truth = {0: 'D06F', 1: 'F25D', 2: 'H04M'}
    # truth = {0: 'H04M', 1: 'F25D', 2: 'D06F'}
    # truth = {0: 'F25D', 1: 'H04M',  2: 'D06F'}
    #truth = {0: 'D06F', 1: 'H04M', 2: 'F25D'}
    # truth = {0: 'F25D', 1: 'D06F', 2: 'H04M'}
    # truth = {0: 'D06F', 1: 'F25D', 2: 'F24F'}
    truth = {0: 'H04M', 1: 'D06F', 2: 'F25D'}
    # truth = {0: 'F24F', 1: 'D06F', 2: 'F25D'}
    # truth = {0: 'F25D', 1: 'F24F', 2: 'D06F'}
    # truth = {0: 'D06F', 1: 'F24F', 2: 'F25D'}
    error = 0.0
    for label in truth:
        for label_ipc in my_ipc[label]:
            if not re.search(truth[label], label_ipc):
                error += 1
    print('聚类准确率为：%f%%' % (100-error/ipc_num*100))

