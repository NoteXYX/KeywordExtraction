# KeywordExtraction
专利关键字提取实验

1、将词向量文件all_rm_abstract_100_mincount1.vec放入 data/word2vec文件夹中。

2、将所有JSON文件全部放入 data/resultJson文件夹中

3、运行dataPre.py，在data/cluster/下生成birch聚类训练文本jsonBirchTrain.txt；在data/test下生成关键词提取测试文本jsonTest.txt

4、运行birchZH.py，在data/figs/下生成聚类结果图JSONcluster.png
