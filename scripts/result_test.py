import re
import operator
from birchZH import get_stopwords
import jieba
import xlwt
import xlrd
import xlutils.copy


def get_test_result(test_name, test_num=100):         #获得各类算法的关键字结果，返回一个字典
    freq_dict = dict()
    tfidf_dict = dict()
    textRank_dict = dict()
    our_dict = dict()
    num = 0
    ifwrite = False
    test_file = open(test_name, 'r', encoding='utf-8')
    test_lines = test_file.readlines()
    for test_line in test_lines:
        if test_line == 'frequency----TF-IDF----textrank----ours-----------------\n':
            ifwrite = True
            num += 1
            if num > test_num:
                break
            freq_keyword = list()
            tfidf_keyword = list()
            textRank_keyword = list()
            our_keyword = list()
        elif test_line == '------------------------------------------------------------------\n':
            ifwrite = False
            freq_dict[num] = freq_keyword
            tfidf_dict[num] = tfidf_keyword
            textRank_dict[num] = textRank_keyword
            our_dict[num] = our_keyword
        else:
            if ifwrite:
                line_split = test_line.split('\t\t\t')
                freq_keyword.append(line_split[0])
                tfidf_keyword.append(line_split[1])
                textRank_keyword.append(line_split[2])
                our_keyword.append(line_split[3].strip())
    test_file.close()
    return freq_dict, tfidf_dict, textRank_dict, our_dict

def get_truth_result(truth_name, get_num=100):       #获得人工标注的关键字结果，返回一个字典
    truth_file = open(truth_name, 'r', encoding = 'utf-8')
    truth_lines = truth_file.readlines()
    truth_dict = dict()
    num = 0
    for truth_line in truth_lines:
        if re.search('keywords:', truth_line):
            num += 1
            if num > get_num:
                break
            keywords = list()
            line_words = truth_line.split('keywords:')[1]
            for word in line_words.split('，'):
                if word.strip() != '' and len(word) > 1:
                    keywords.append(word.strip())
            truth_dict[num] = keywords
            print('第%d条人工标注专利关键字提取完成......' % num)
        elif re.search('keywords: ',truth_line):
            num += 1
            if num > get_num:
                break
            keywords = list()
            line_words = truth_line.split('keywords: ')[1]
            for word in line_words.split('，'):
                if word.strip() != '' and len(word) > 1:
                    keywords.append(word.strip())
            truth_dict[num] = keywords
            print('第%d条人工标注专利关键字提取完成......' % num)
    truth_file.close()
    return truth_dict

def acc_test(truth_name, test_name, truth_top_k=10, test_top_k=10):
    freq_dict, tfidf_dict, textRank_dict,  our_dict = get_test_result(test_name)
    truth_dict = get_truth_result(truth_name)
    freq_true_num = 0.0
    tfidf_true_num = 0.0
    textRank_true_num = 0.0
    our_true_num = 0.0
    truth_num = 0.0
    test_num = 0.0
    for patent_index in truth_dict:
        truth_keywords = truth_dict[patent_index]
        test_truth_keywords = truth_keywords[0: min(truth_top_k, len(truth_keywords)): 1]
        freq_keywords = freq_dict[patent_index]
        test_freq_keywords = freq_keywords[0: min(test_top_k, len(freq_keywords)): 1]
        tfidf_keywords = tfidf_dict[patent_index]
        test_tfidf_keywords = tfidf_keywords[0: min(test_top_k, len(tfidf_keywords)): 1]
        textRank_keywords = textRank_dict[patent_index]
        test_textRank_keywords = textRank_keywords[0 : min(test_top_k, len(textRank_keywords)) : 1]
        our_keywords = our_dict[patent_index]
        test_our_keywords = our_keywords[0 : min(test_top_k, len(our_keywords)) : 1]
        assert len(test_freq_keywords) == len(test_tfidf_keywords) == len(test_textRank_keywords) == len(test_our_keywords)
        #   acc
        cur_test_num = min(test_top_k, len(test_our_keywords))
        test_num += cur_test_num    #####################
        for test_keyword_index in range(cur_test_num):  ################
            if test_freq_keywords[test_keyword_index] in test_truth_keywords:
                freq_true_num += 1
            if test_tfidf_keywords[test_keyword_index] in test_truth_keywords:
                tfidf_true_num += 1
            if test_textRank_keywords[test_keyword_index] in test_truth_keywords:
                textRank_true_num += 1
            if test_our_keywords[test_keyword_index] in test_truth_keywords:
                our_true_num += 1
        #   recall
        # cur_truth_num = min(truth_top_k, len(truth_keywords))
        # truth_num += cur_truth_num    #####################
        # for truth_keyword_index in range(cur_truth_num):  ################
        #     truth_keyword = truth_keywords[truth_keyword_index]
        #     if truth_keyword in test_freq_keywords:
        #         freq_true_num += 1
        #     if truth_keyword in test_tfidf_keywords:
        #         tfidf_true_num += 1
        #     if truth_keyword in test_textRank_keywords:
        #         textRank_true_num += 1
        #     if truth_keyword in test_our_keywords:
        #         our_true_num += 1
    freq_acc = freq_true_num / test_num * 100
    tfidf_acc = tfidf_true_num / test_num * 100
    textRank_acc = textRank_true_num / test_num * 100
    our_acc = our_true_num / test_num * 100
    # freq_recall = freq_true_num / truth_num * 100
    # tfidf_recall = tfidf_true_num / truth_num * 100
    # textRank_recall = textRank_true_num / truth_num * 100
    # our_recall = our_true_num / truth_num * 100
    print('frequency准确率为：%f%%' % freq_acc)
    print('TF-IDF准确率为：%f%%' % tfidf_acc)
    print('textRank准确率为：%f%%' % textRank_acc)
    print('our准确率为：%f%%' % our_acc)
    return freq_acc, tfidf_acc, textRank_acc, our_acc
    # return freq_recall, tfidf_recall, textRank_recall, our_recall

def main():
    truth_name = r'..\data\patent_abstract\6种专利摘要各100条已标注\空调植文武.txt'
    test_name = r'..\data\patent_abstract\6种专利摘要各100条已标注\kongtiao_freq_TFIDF_textRank_ours_techField_wordAVG_1.009_50.txt'
    test_top_k = 20
    truth_top_k = 5
    name_index = 2
    if name_index == 1:  #第一个人为2，第二个人为6
        name = 2
    elif name_index == 2:
        name = 6
    freq_acc, tfidf_acc, textRank_acc, our_acc = acc_test(truth_name, test_name, truth_top_k=truth_top_k, test_top_k=test_top_k)
    data = xlrd.open_workbook(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\实验结果.xls')
    ws = xlutils.copy.copy(data)
    table = ws.get_sheet(0)
    title_line_num = 0
    title_line_xishu = 0
    if re.search('电视', truth_name):
        title_line_xishu = 1
    if re.search('清洁', truth_name):
        title_line_xishu = 2
    if re.search('冰箱', truth_name):
        title_line_xishu = 3
    if re.search('洗衣机', truth_name):
        title_line_xishu = 4
    if re.search('移动通信', truth_name):
        title_line_xishu = 5
    title_line_num += title_line_xishu * 16
    write_line_num = title_line_num + name + test_top_k/5
    if truth_top_k == 5:
        table.write(write_line_num, 2, '%.2f' % freq_acc)
        table.write(write_line_num, 4, '%.2f' % tfidf_acc)
        table.write(write_line_num, 6, '%.2f' % textRank_acc)
        table.write(write_line_num, 8, '%.2f' % our_acc)
    if truth_top_k == 10:
        table.write(write_line_num, 3, '%.2f' % freq_acc)
        table.write(write_line_num, 5, '%.2f' % tfidf_acc)
        table.write(write_line_num, 7, '%.2f' % textRank_acc)
        table.write(write_line_num, 9, '%.2f' % our_acc)

    ws.save(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\实验结果.xls')



if __name__ == '__main__':
    main()