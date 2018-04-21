# -*- coding:UTF-8 -*-
"""
@author: ZhaoHe
"""
import os
from src.config import host, port, user, passwd, dbname, dataset2_table_name
from src.association import Association
import pymysql as mdb
import json

class Data(object):
    def __init__(self):
        # 连接到数据库
        self.conn = mdb.connect(host=host, port=port, user=user, passwd=passwd, \
                                      db=dbname, charset='utf8')
        self.cursor = self.conn.cursor()

        # 属性列名文件路径
        self.dataset2_nom_col_path = r'./data/dataset2/关联分析属性列名.txt'

        # 属性名列表
        self.dataset2_nom_feature_list = self.get_feature_list(self.dataset2_nom_col_path)

        # 结果文件路径
        self.result_path = r'./results'


    def get_feature_list(self, column_file_path):
        """
        获取全部标称/数值属性名列表
        :param column_file_path: 存有标称/数值属性列名的.txt文件，每行为一个属性名
        :return: 属性名列表
        """
        feature_list = []
        column_file = open(column_file_path, 'r')
        for line in column_file:
            feature_name = line.strip()
            feature_name = "_".join(feature_name.split(' '))
            feature_list.append(feature_name)
        return feature_list

    def get_feature_values(self, feature_name, table_name):
        """
        对于某一个属性feature_name, 在数据库中查询，返回该列对应所有值
        :param feature_name: 属性名
        :param table_name: 数据集对应的表名
        :return: 值列表
        """
        values = [feature_name]
        sql = "SELECT %s FROM %s" % (feature_name, table_name)
        self.cursor.execute(sql)
        for value in self.cursor.fetchall():
            values.append(value[0])
        return values

    def process_nom_features(self, feature_list, table_name):
        """
        处理一批标称属性，获取所有可能的取值及其对应个数（包括缺失值的个数）
        :param feature_list: 属性列表
        :param table_name: 数据集对应的表名
        :return: 一个字典，key为所有可能取值，value为取值对应的个数
        """
        table_name = dataset2_table_name
        out_path = self.result_path
        association = Association()

        # 遍历全部属性
        columns = []
        for feature_name in feature_list:
            print("Dealing with feature: {}".format(feature_name))
            feature_col = self.get_feature_values(feature_name, table_name)
            columns.append(feature_col)
        rows = list(zip(*columns))
        # 将数据转为数据字典存储
        dataset = []
        feature_names = rows[0]
        for data_line in rows[1:]:
            data_set = []
            for i , value in enumerate(data_line):
                if not value:
                    data_set.append((feature_names[i], 'NA'))
                else:
                    data_set.append((feature_names[i], value))
            dataset.append(data_set)

        # 获取频繁项集
        freq_set , support_data = association.apriori(dataset)
        support_data_out = sorted(support_data.items(), key= lambda d:d[1],reverse=True)
        print(support_data)
        # 获取强关联规则列表
        big_rules_list = association.generate_rules(freq_set, support_data)
        big_rules_list = sorted(big_rules_list, key= lambda x:x[3], reverse=True)
        print(big_rules_list)

        # 将频繁项集输出到结果文件
        freq_set_file = open(os.path.join(out_path, 'freq_set.json'), 'w')
        for (key, value) in support_data_out:
            result_dict = {'set':None, 'sup':None}
            set_result = list(key)
            sup_result = value
            result_dict['set'] = set_result
            result_dict['sup'] = sup_result
            json_str = json.dumps(result_dict, ensure_ascii=False)
            freq_set_file.write(json_str+'\n')
        freq_set_file.close()

        # 将关联规则输出到结果文件
        rules_file = open(os.path.join(out_path, 'rules.json'), 'w')
        for result in big_rules_list:
            result_dict = {'X_set':None, 'Y_set':None, 'sup':None, 'conf':None, 'lift':None}
            X_set, Y_set, sup, conf, lift = result
            result_dict['X_set'] = list(X_set)
            result_dict['Y_set'] = list(Y_set)
            result_dict['sup'] = sup
            result_dict['conf'] = conf
            result_dict['lift'] = lift
            json_str = json.dumps(result_dict, ensure_ascii=False)
            rules_file.write(json_str + '\n')
        rules_file.close()




