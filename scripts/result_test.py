import re
import operator
from birchZH import get_stopwords
import jieba



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

def acc_test(truth_name, test_name, top_k=10):
    freq_dict, tfidf_dict, textRank_dict,  our_dict = get_test_result(test_name)
    truth_dict = get_truth_result(truth_name)
    freq_true_num = 0.0
    tfidf_true_num = 0.0
    textRank_true_num = 0.0
    our_true_num = 0.0
    truth_num = 0.0
    for patent_index in truth_dict:
        truth_keywords = truth_dict[patent_index]
        freq_keywords = freq_dict[patent_index]
        test_freq_keywords = freq_keywords[0: min(top_k, len(freq_keywords)): 1]
        tfidf_keywords = tfidf_dict[patent_index]
        test_tfidf_keywords = tfidf_keywords[0: min(top_k, len(tfidf_keywords)): 1]
        textRank_keywords = textRank_dict[patent_index]
        test_textRank_keywords = textRank_keywords[0 : min(top_k, len(textRank_keywords)) : 1]
        our_keywords = our_dict[patent_index]
        test_our_keywords = our_keywords[0 : min(top_k, len(our_keywords)) : 1]
        truth_num += len(truth_keywords)
        for truth_keyword_index in range(len(truth_keywords)):
            truth_keyword = truth_keywords[truth_keyword_index]
            if truth_keyword in test_freq_keywords:
                freq_true_num += 1
            if truth_keyword in test_tfidf_keywords:
                tfidf_true_num += 1
            if truth_keyword in test_textRank_keywords:
                textRank_true_num += 1
            if truth_keyword in test_our_keywords:
                our_true_num += 1
    print('frequency准确率为：%f%%' % (freq_true_num / truth_num * 100))
    print('TF-IDF准确率为：%f%%' % (tfidf_true_num / truth_num * 100))
    print('textRank准确率为：%f%%' % (textRank_true_num / truth_num * 100))
    print('our准确率为：%f%%' % (our_true_num / truth_num * 100))

def main():
    truth_name = r'..\data\patent_abstract\6种专利摘要各100条已标注\洗衣机李玉玲.txt'
    test_name = r'..\data\patent_abstract\6种专利摘要各100条已标注\xiyiji_freq_TFIDF_textRank_ours_techField_wordAVG_1.04_50.txt'
    acc_test(truth_name, test_name, top_k=10)

if __name__ == '__main__':
    main()