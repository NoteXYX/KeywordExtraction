import pymysql
import sys
import io
import codecs
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码
if __name__ == '__main__':
    db = pymysql.connect("localhost", "root", "", "patent_system")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    log_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\_kTVq_techField.txt', 'w', encoding='utf-8')
    sql = """ SELECT label, tech_field FROM tb_patentall_label where (label LIKE '%F24F%') OR (label LIKE '%H04N%') OR (label LIKE '%B08B%'); """
    num0 = 0
    num1 = 0
    num2 = 0
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        # 获取所有记录列表
        results = cursor.fetchall()
        i = 0
        for row in results:
            if re.search('F24F', row[0]) and num0 < 1000:
                num0 += 1
                # print(row[0] + ' ::  ' + row[1])
                # log_file.write('%s ::  %s\n' % (row[0], row[1]))
                print(row[1])
                log_file.write('%s\n' % row[1])
                i += 1
                print("第%d条专利成功写入文档!" % i)
                print('************************************************************************')
            if re.search('H04N', row[0]) and num1 < 1000:
                num1 += 1
                # print(row[0] + ' ::  ' + row[1])
                # log_file.write('%s ::  %s\n' % (row[0], row[1]))
                print(row[1])
                log_file.write('%s\n' % row[1])
                i += 1
                print("第%d条专利成功写入文档!" % i)
                print('************************************************************************')
            if re.search('B08B', row[0]) and num2 < 1000:
                num2 += 1
                # print(row[0] + ' ::  ' + row[1])
                # log_file.write('%s ::  %s\n' % (row[0], row[1]))
                print(row[1])
                log_file.write('%s\n' % row[1])
                i += 1
                print("第%d条专利成功写入文档!" % i)
                print('************************************************************************')
        # for row in results:      #YDY
        #     if num0 == 200:
        #         break
        #     else:
        #         print(row[0] + ' ::  ' + row[1])
        #         if re.search('F24F', row[0]):
        #             cur_label = 0
        #             log_file.write('%s ::  %s\n' % (cur_label, row[1]))
        #             num0 +=1
        #             i += 1
        #         print("第%d条专利成功写入文档!" % i)
        #         print('************************************************************************')
        # for row in results:
        #     if num1 == 200:
        #         break
        #     else:
        #         print(row[0] + ' ::  ' + row[1])
        #         if re.search('H04N', row[0]):
        #             cur_label = 1
        #             log_file.write('%s ::  %s\n' % (cur_label, row[1]))
        #             num1 += 1
        #             i += 1
        db.commit()
    except IndexError as e:
        # 如果发生错误则回滚
        db.rollback()
        print(e)



    # 关闭数据库连接
    db.close()
    log_file.close()
    print('num0:' + str(num0))
    print('num1:' + str(num1))
    print('num2:' + str(num2))
