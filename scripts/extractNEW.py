import pymysql
import sys
import io
import codecs


sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码
if __name__ == '__main__':
    db = pymysql.connect("localhost", "root", "", "patent_system")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    sql = """ SELECT count(id) FROM tb_patentall_label; """
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        # 获取所有记录列表
        results = cursor.fetchall()
        patent_num = results[0][0]
        print('专利数目：' + str(patent_num) )
        db.commit()
    except IndexError as e:
        # 如果发生错误则回滚
        db.rollback()
        print(e)
    log_file = open(r'..\data\patent_abstract\bxkbxk.txt', 'w', encoding='utf-8')
    sql = """ SELECT abstract FROM tb_patent;  """
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        # 获取所有记录列表
        results = cursor.fetchall()
        i = 0
        for row in results:
            print(row[0])
            log_file.write('%s\n' % (row[0]))
            # log_file.write('-------keyword-------\n')
            # tr4w = TextRank4Keyword(stop_words_file = '../textrank4zh/stopwords.txt')
            # tr4w.analyze(text=row[0], lower=True, vertex_source = 'no_stop_words', window=3, pagerank_config={'alpha': 0.85})
            # for item in tr4w.get_keywords(20, word_min_len=2):
            #     log_file.write('%s\t%f\n' % (item.word, item.weight))
            # log_file.write("-------phrase-------\n")
            # for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num=0):
            #     log_file.write('%s\n' % (phrase))
            i+=1
            print("第%d条专利成功写入文档!" % i)
            print('************************************************************************')
        db.commit()
    except IndexError as e:
        # 如果发生错误则回滚
        db.rollback()
        print(e)


    # 关闭数据库连接
    db.close()
    log_file.close()
