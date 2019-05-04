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

def get_truth_result(truth_name):       #获得人工标注的关键字结果，返回一个字典
    truth_file = open(truth_name, 'r', encoding = 'utf-8')
    truth_lines = truth_file.readlines()
    for truth_line in truth_lines:
        if re.search(' ::  ', truth_line):

    truth_file.close()

def main():
    truth_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\6种专利摘要各100条已标注\空调谢育欣.txt'
    test_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\6种专利摘要各100条已标注\kongtiao_textRankVSours_techField_wordAVG_1.009_50.txt'
    textRank_dict,  our_dict = get_test_result(test_name)


if __name__ == '__main__':
    main()
