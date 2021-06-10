import json
import os
import argparse

def search(folder, filters, allFiles):      # 搜索目录下的所有文件
    folders = os.listdir(folder)
    for name in folders:
        curname = os.path.join(folder, name)
        isfile = os.path.isfile(curname)
        if isfile:
            for filter in filters:
                if name.endswith(filter):
                    allFiles.append(curname)
                    break
        else:
            search(curname, filters, allFiles)
    return allFiles

def main(jsonDir, birchTrain, testName):
    filters = ['.json', '.JSON']
    allFiles = list()
    allFiles = search(jsonDir, filters, allFiles)
    birchTrainFile = open(birchTrain, 'w', encoding='utf-8')
    keywordExTestFile = open(testName, 'w', encoding='utf-8')
    fileNum = 0
    for jsonName in allFiles:
        fileNum += 1
        print('处理第%d个json文件......' % fileNum)
        with open(jsonName, 'r', encoding='utf-8') as load_f:
            load_dict = json.load(load_f)
            muluTxt = load_dict['目录文本']
            label = load_dict['人物简介']['name']
            lenOfTxt = len(muluTxt)
            i = 0
            while i < lenOfTxt*2//3:
                birchTrainFile.write('%s ::  %s\n' % (label, muluTxt[i]['text']))
                i += 1
            while i < lenOfTxt:
                keywordExTestFile.write('%s ::  %s\n' % (label, muluTxt[i]['text']))
                i += 1
    birchTrainFile.close()
    keywordExTestFile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UPKEM')
    parser.add_argument('--jsonDir', '-j', help='JSON数据所在文件夹',
                        default=r'../data/resultJson')
    parser.add_argument('--birchTrain', '-b', help='生成的birch聚类训练文本',
                        default=r'../data/cluster/jsonBirchTrain.txt')
    parser.add_argument('--testName', '-t', help='生成的关键词提取测试文本',
                        default=r'../data/test/jsonTest.txt')
    args = parser.parse_args()
    jsonDir = args.jsonDir
    birchTrain = args.birchTrain
    testName = args.testName
    main(jsonDir, birchTrain, testName)