import re
import operator
from birchZH import get_stopwords
import jieba
import xlwt
import xlrd
import xlutils.copy


def get_test_result(test_name, test_num=100):         #获得各类算法的关键字结果，返回一个字典
    rake_dict = dict()
    tfidf_dict = dict()
    textRank_dict = dict()
    PKEA_dict = dict()
    our_dict = dict()
    num = 0
    ifwrite = False
    test_file = open(test_name, 'r', encoding='utf-8')
    test_lines = test_file.readlines()
    for test_line in test_lines:
        if test_line == 'RAKE----TF-IDF----textrank----PKEA----ours-----------------\n':
            ifwrite = True
            num += 1
            if num > test_num:
                break
            rake_keywords = list()
            tfidf_keywords = list()
            textRank_keywords = list()
            PKEA_keywords = list()
            our_keywords = list()
        elif test_line == '------------------------------------------------------------------\n':
            ifwrite = False
            rake_dict[num] = rake_keywords
            tfidf_dict[num] = tfidf_keywords
            textRank_dict[num] = textRank_keywords
            PKEA_dict[num] = PKEA_keywords
            our_dict[num] = our_keywords
        else:
            if ifwrite:
                line_split = test_line.split('\t\t')
                rake_keywords.append(line_split[0])
                tfidf_keywords.append(line_split[1])
                textRank_keywords.append(line_split[2])
                PKEA_keywords.append(line_split[3])
                our_keywords.append(line_split[4].strip())
    test_file.close()
    return rake_dict, tfidf_dict, textRank_dict, PKEA_dict, our_dict

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
            for word in line_words.split('、'):
                if word.strip() != '' and len(word) > 1:
                    keywords.append(word.strip())
            truth_dict[num] = keywords
            print('第%d条人工标注专利关键字提取完成......' % num)
    truth_file.close()
    return truth_dict

def result_test(truth_name, test_name, test_model, truth_top_k=10, test_top_k=10):
    rake_dict, tfidf_dict, textRank_dict, PKEA_dict, our_dict = get_test_result(test_name)
    truth_dict = get_truth_result(truth_name)
    rake_acc_true_num, rake_recall_true_num = 0.0, 0.0
    tfidf_acc_true_num, tfidf_recall_true_num = 0.0, 0.0
    textRank_acc_true_num, textRank_recall_true_num = 0.0, 0.0
    PKEA_acc_true_num, PKEA_recall_true_num = 0.0, 0.0
    our_acc_true_num, our_recall_true_num = 0.0, 0.0
    test_num = 0.0
    truth_num = 0.0
    for patent_index in truth_dict:
        truth_keywords = truth_dict[patent_index]
        test_truth_keywords = truth_keywords[0: min(truth_top_k, len(truth_keywords)): 1]
        rake_keywords = rake_dict[patent_index]
        test_rake_keywords = rake_keywords[0: min(test_top_k, len(rake_keywords)): 1]
        tfidf_keywords = tfidf_dict[patent_index]
        test_tfidf_keywords = tfidf_keywords[0: min(test_top_k, len(tfidf_keywords)): 1]
        textRank_keywords = textRank_dict[patent_index]
        test_textRank_keywords = textRank_keywords[0 : min(test_top_k, len(textRank_keywords)) : 1]
        PKEA_keywords = PKEA_dict[patent_index]
        test_PKEA_keywords = PKEA_keywords[0: min(test_top_k, len(PKEA_keywords)): 1]
        our_keywords = our_dict[patent_index]
        test_our_keywords = our_keywords[0 : min(test_top_k, len(our_keywords)) : 1]
        assert len(test_rake_keywords) == len(test_tfidf_keywords) == len(test_textRank_keywords) == len(test_PKEA_keywords) == len(test_our_keywords)
        # acc
        cur_test_num = min(test_top_k, len(test_our_keywords))
        test_num += cur_test_num
        for test_keyword_index in range(cur_test_num):
            if test_rake_keywords[test_keyword_index] in test_truth_keywords:
                rake_acc_true_num += 1
            if test_tfidf_keywords[test_keyword_index] in test_truth_keywords:
                tfidf_acc_true_num += 1
            if test_textRank_keywords[test_keyword_index] in test_truth_keywords:
                textRank_acc_true_num += 1
            if test_PKEA_keywords[test_keyword_index] in test_truth_keywords:
                PKEA_acc_true_num += 1
            if test_our_keywords[test_keyword_index] in test_truth_keywords:
                our_acc_true_num += 1
        # recall
        cur_truth_num = min(truth_top_k, len(truth_keywords))
        truth_num += cur_truth_num
        for truth_keyword_index in range(cur_truth_num):
            truth_keyword = truth_keywords[truth_keyword_index]
            if truth_keyword in test_rake_keywords:
                rake_recall_true_num += 1
            if truth_keyword in test_tfidf_keywords:
                tfidf_recall_true_num += 1
            if truth_keyword in test_textRank_keywords:
                textRank_recall_true_num += 1
            if truth_keyword in test_PKEA_keywords:
                PKEA_recall_true_num += 1
            if truth_keyword in test_our_keywords:
                our_recall_true_num += 1
    # acc
    rake_acc = rake_acc_true_num / test_num * 100
    tfidf_acc = tfidf_acc_true_num / test_num * 100
    textRank_acc = textRank_acc_true_num / test_num * 100
    PKEA_acc = PKEA_acc_true_num / test_num * 100
    our_acc = our_acc_true_num / test_num * 100
    # recall
    rake_recall = rake_recall_true_num / truth_num * 100
    tfidf_recall = tfidf_recall_true_num / truth_num * 100
    textRank_recall = textRank_recall_true_num / truth_num * 100
    PKEA_recall = PKEA_recall_true_num / truth_num * 100
    our_recall = our_recall_true_num / truth_num * 100
    if test_model == 'acc':
        print('RAKE准确率为：%f%%' % rake_acc)
        print('TF-IDF准确率为：%f%%' % tfidf_acc)
        print('textRank准确率为：%f%%' % textRank_acc)
        print('PKEA准确率为：%f%%' % PKEA_acc)
        print('our准确率为：%f%%' % our_acc)
        return rake_acc, tfidf_acc, textRank_acc, PKEA_acc, our_acc
    elif test_model == 'recall':
        print('RAKE召回率为：%f%%' % rake_recall)
        print('TF-IDF召回率为：%f%%' % tfidf_recall)
        print('textRank召回率为：%f%%' % textRank_recall)
        print('PKEA召回率为：%f%%' % PKEA_recall)
        print('our召回率为：%f%%' % our_recall)
        return rake_recall, tfidf_recall, textRank_recall, PKEA_recall, our_recall
    elif test_model == 'F1':
        rake_F1 = (2 * rake_acc * rake_recall) / (rake_acc + rake_recall)
        tfidf_F1 = (2 * tfidf_acc * tfidf_recall) / (tfidf_acc + tfidf_recall)
        textRank_F1 = (2 * textRank_acc * textRank_recall) / (textRank_acc + textRank_recall)
        PKEA_F1 = (2 * PKEA_acc * PKEA_recall) / (PKEA_acc + PKEA_recall)
        our_F1 = (2 * our_acc * our_recall) / (our_acc + our_recall)
        print('RAKE的F1值为：%f%%' % rake_F1)
        print('TF-IDF的F1值为：%f%%' % tfidf_F1)
        print('textRank的F1值为：%f%%' % textRank_F1)
        print('PKEA的F1值为：%f%%' %PKEA_F1)
        print('our的F1值为：%f%%' % our_F1)
        return rake_F1, tfidf_F1, textRank_F1, PKEA_F1, our_F1

def main():
    truth_name = r'..\data\patent_abstract\6种专利摘要各100条已标注\空调综合.txt'
    test_name = r'..\data\patent_abstract\6种专利摘要各100条已标注\kongtiao_RAKE_TFIDF_textRank_PKEA_ours_techField_wordAVG_1.0115_50.txt'
    test_model = 'F1'
    test_top_k = 5
    truth_top_k = 10
    # name_index = 1
    # if re.search('植文武', truth_name) or re.search('丁晗', truth_name) or re.search('唐雪涛', truth_name) or re.search('岳永政', truth_name):
    #     name_index = 2
    # elif re.search('综合', truth_name):
    #     name_index = 3
    # name = name_index - 1
    rake_result, tfidf_result, textRank_result, PKEA_result, our_result = result_test(truth_name, test_name, test_model, truth_top_k=truth_top_k,test_top_k=test_top_k)
    # data = xlrd.open_workbook(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\truth_top10_F1实验结果.xls')
    # ws = xlutils.copy.copy(data)
    # table = ws.get_sheet(0)
    # title_line_num = 0
    # title_line_xishu = 0
    # if re.search('电视', truth_name):
    #     title_line_xishu = 1
    # if re.search('清洁', truth_name):
    #     title_line_xishu = 2
    # if re.search('冰箱', truth_name):
    #     title_line_xishu = 3
    # if re.search('洗衣机', truth_name):
    #     title_line_xishu = 4
    # if re.search('移动通信', truth_name):
    #     title_line_xishu = 5
    # title_line_num += title_line_xishu * 8
    # write_col_num = int(test_top_k / 5)
    # table.write(title_line_num + 3, write_col_num, '%.2f' % rake_result)
    # table.write(title_line_num + 4, write_col_num, '%.2f' % tfidf_result)
    # table.write(title_line_num + 5, write_col_num, '%.2f' % textRank_result)
    # table.write(title_line_num + 6, write_col_num, '%.2f' % our_result)
    #
    # ws.save(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\truth_top10_F1实验结果.xls')



if __name__ == '__main__':
    main()