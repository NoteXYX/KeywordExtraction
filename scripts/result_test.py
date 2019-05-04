import re


def get_test_result(test_name, test_num=100):         #获得各类算法的关键字结果，返回一个字典
    textRank_dict = dict()
    our_dict = dict()
    num = 0
    ifwrite = False
    test_file = open(test_name, 'r', encoding='utf-8')
    test_lines = test_file.readlines()
    for test_line in test_lines:
        if test_line == 'textrank----ours-----------------\n':
            ifwrite = True
            num += 1
            if num > test_num:
                break
            textRank_keyword = list()
            our_keyword = list()
        elif test_line == '------------------------------------------------------------------\n':
            ifwrite = False
            textRank_dict[num] = textRank_keyword
            our_dict[num] = our_keyword
        else:
            if ifwrite:
                line_split = test_line.split('\t\t\t')
                textRank_keyword.append(line_split[0])
                our_keyword.append(line_split[1].strip())
    test_file.close()
    return textRank_dict, our_dict

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
                if word.strip() != '':
                    keywords.append(word.strip())
            truth_dict[num] = keywords
            print('第%d条人工标注专利关键字提取完成......' % num)
        elif re.search('keywords: ',truth_line):
            num += 1
            if num > get_num:
                break
            keywords = list()
            line_words = truth_line.split('keywords: ')[1]
            for word in line_words.split('、'):
                if word.strip() != '':
                    keywords.append(word.strip())
            truth_dict[num] = keywords
            print('第%d条人工标注专利关键字提取完成......' % num)
    truth_file.close()
    return truth_dict

def main():
    truth_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\6种专利摘要各100条已标注\电视余道远.txt'
    test_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\6种专利摘要各100条已标注\TV_textRankVSours_techField_wordAVG_1.009_50.txt'
    top_k = 20
    textRank_dict,  our_dict = get_test_result(test_name)
    truth_dict = get_truth_result(truth_name)
    textRank_true_num = 0.0
    our_true_num = 0.0
    truth_num = 0.0
    for patent_index in truth_dict:
        truth_keywords = truth_dict[patent_index]
        textRank_keywords = textRank_dict[patent_index]
        test_textRank_keywords = textRank_keywords[0 : min(top_k, len(textRank_keywords)) : 1]
        our_keywords = our_dict[patent_index]
        test_our_keywords = our_keywords[0 : min(top_k, len(our_keywords)) : 1]
        truth_num += len(truth_keywords)
        for truth_keyword_index in range(len(truth_keywords)):
            truth_keyword = truth_keywords[truth_keyword_index]
            if truth_keyword in test_textRank_keywords:
                textRank_true_num += 1
            if truth_keyword in test_our_keywords:
                our_true_num += 1

    print('textRank准确率为：%f%%' % (textRank_true_num / truth_num * 100))
    print('our准确率为：%f%%' % (our_true_num / truth_num * 100))



if __name__ == '__main__':
    main()
