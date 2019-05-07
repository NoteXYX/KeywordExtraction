import re


def get_truth_result(truth_name, get_num=100):       #获得人工标注的关键字结果，返回一个字典{1:[key1,key2...]}
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
    truth_file1 = open(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注\冰箱植文武old.txt', 'r', encoding='utf-8')
    truth_file2 = open(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注\冰箱植文武.txt', 'w', encoding='utf-8')
    truth_lines = truth_file1.readlines()
    for truth_line in truth_lines:
        if re.search('keywords:', truth_line):
            keywords = list()
            line_words = truth_line.split('keywords:')[1]
            for word in line_words.split('，'):
                if word.strip() != '':
                    keywords.append(word.strip())
            truth_file2.write('keywords:' + '、'.join(keywords) + '\n')
        else:
            truth_file2.write(truth_line)
    truth_file1.close()
    truth_file2.close()


if __name__ == '__main__':
    main()