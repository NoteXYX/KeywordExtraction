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

def jiao_truth(truth_dict1, truth_dict2):
    truth_zonghe = dict()
    assert len(truth_dict1) == len(truth_dict2)
    for num in truth_dict1:
        keywords_zonghe = list(set(truth_dict1[num]).intersection(set(truth_dict2[num])))
        truth_zonghe[num] = keywords_zonghe
    return truth_zonghe

def main():
    truth_name1 = r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注\空调谢育欣.txt'
    truth_name2 = r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注\空调植文武.txt'
    truth_file = open(truth_name1, 'r',encoding='utf-8')
    file_zonghe = open(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注\空调综合.txt', 'w', encoding='utf-8')
    truth_dict1 = get_truth_result(truth_name1)
    truth_dict2 = get_truth_result(truth_name2)
    truth_zonghe = jiao_truth(truth_dict1, truth_dict2)
    truth_lines = truth_file.readlines()
    patent_num = 1
    for truth_line in truth_lines:
        if re.search('keywords:', truth_line):
            file_zonghe.write('keywords:' + '、'.join(truth_zonghe[patent_num]) + '\n')
            patent_num += 1
        else:
            file_zonghe.write(truth_line)
    truth_file.close()
    file_zonghe.close()

if __name__ == '__main__':
    main()