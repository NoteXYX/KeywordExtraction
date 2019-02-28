import jieba

try:
    f1 = open('../data/patent_abstract/_all_rm_abstract.txt', 'r', encoding='utf-8')
    f2 = open('../data/patent_abstract/all_fc_abstract.txt', 'w', encoding='utf-8')
    mystr = f1.readlines()
    iters = 1
    for line_str in mystr:
        seg_list = jieba.cut(line_str)
        result = ' '.join(seg_list)
        f2.write(result)
        print('处理完成%d行'%(iters))
        iters+=1
    #print(mystr[0])

finally:
    if f1:
        f1.close()
    if f2:
        f2.close()
