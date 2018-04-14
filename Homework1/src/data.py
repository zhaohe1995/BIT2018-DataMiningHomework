# -*- coding:UTF-8 -*-
"""
@author: ZhaoHe
"""
import os
from src.config import host, port, user, passwd, dbname, dataset1_table_name, dataset2_table_name, impute_feature_list
from src.num_processor import NumProcessor
import pymysql as mdb

class Data(object):
    def __init__(self):
        # 连接到数据库
        self.conn = mdb.connect(host=host, port=port, user=user, passwd=passwd, \
                                      db=dbname, charset='utf8')
        self.cursor = self.conn.cursor()

        # 属性列名文件路径
        self.dataset1_nom_col_path = r'../data/dataset1/标称属性列名.txt'
        self.dataset1_num_col_path = r'../data/dataset1/数值属性列名.txt'
        self.dataset2_nom_col_path = r'../data/dataset2/标称属性列名.txt'
        self.dataset2_num_col_path = r'../data/dataset2/数值属性列名.txt'

        # 属性名列表
        self.dataset1_nom_feature_list = self.get_feature_list(self.dataset1_nom_col_path)
        self.dataset1_num_feature_list = self.get_feature_list(self.dataset1_num_col_path)
        self.dataset2_nom_feature_list = self.get_feature_list(self.dataset2_nom_col_path)
        self.dataset2_num_feature_list = self.get_feature_list(self.dataset2_num_col_path)

        # 结果文件路径
        self.result_path = r'../results'


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
        values = []
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
        # 设置输出结果文件路径
        if table_name == dataset1_table_name:
            out_path = os.path.join(self.result_path, 'dataset1', '标称属性')
        elif table_name == dataset2_table_name:
            out_path = os.path.join(self.result_path, 'dataset2', '标称属性')
        else:
            raise FilenameError("Please check the dataset name!")

        # 遍历全部属性
        for feature_name in feature_list:
            value_dict = dict()
            value_num = 0
            values = self.get_feature_values(feature_name, table_name)
            for value in values:
                if value not in value_dict:
                    value_num += 1
                    value_dict[value] = 1
                else:
                    value_dict[value] += 1
            # 结果输出到文件
            feature_out_path = os.path.join(out_path, feature_name+'.txt')
            feature_out_file = open(feature_out_path, 'wb')
            feature_out_file.write(("Feature Name: %s\n" % feature_name).encode('utf-8'))
            feature_out_file.write(("Value Num: %s\n" % value_num).encode('utf-8'))
            for value, freq in value_dict.items():
                if value == None:
                    value = 'Null'
                out_string = value +','+str(freq)+'\n'
                feature_out_file.write(out_string.encode('utf-8'))
            feature_out_file.close()
            # 结果输出到console
            print("-------------------------------------------------------")
            print("Feature Name: %s" % feature_name)
            print("Value Num: %s" % value_num)
            for value, freq in value_dict.items():
                print(value, freq)

    def process_num_features(self, feature_list, table_name):
        """
        （主要用到专用于处理数值属性的类NumProcessor进行处理）
        处理一批数值属性，获取关于数据分布的各个值，并绘制分布的直方图、检验正太分布的qq图、盒图、并识别离群值
        分别用四种方法对缺失值进行处理，对于处理结果进行可视化分析
        :param feature_list: 属性列表
        :param table_name: 数据集对应的表名
        :return: 对应数据分布的属性值、绘图等
        """
        # 设置输出结果文件路径
        if table_name == dataset1_table_name:
            out_path = os.path.join(self.result_path, 'dataset1', '数值属性')
            fig_path = os.path.join(self.result_path, 'dataset1', 'figure')
        elif table_name == dataset2_table_name:
            out_path = os.path.join(self.result_path, 'dataset2', '数值属性')
            fig_path = os.path.join(self.result_path, 'dataset2', 'figure')
        else:
            raise FilenameError("Please check the dataset name!")

        # 遍历全部属性
        for feature_name in feature_list:
            values = self.get_feature_values(feature_name, table_name)
            processor = NumProcessor(feature_name, values)
            # 获取描述数据集的几个数值
            missing_num, max_num, min_num, average_num, median_num, quartile1, quartile2 = processor.pre_process()
            # 为数据画直方图
            processor.draw_histogram(processor.new_value_set, feature_name, fig_path)
            # 为数据画q-q图
            processor.draw_qq_plot(processor.new_value_set, feature_name, fig_path)
            # 为数据画盒图
            processor.draw_box(processor.new_value_set, feature_name, fig_path)

            # 结果输出到文件
            feature_out_path = os.path.join(out_path, feature_name + '.txt')
            feature_out_file = open(feature_out_path, 'wb')
            feature_out_file.write(("Feature Name: %s\n" % feature_name).encode('utf-8'))
            feature_out_file.write(("Missing Num: %s\n" % missing_num).encode('utf-8'))
            feature_out_file.write(("Max Num: %s\n" % max_num).encode('utf-8'))
            feature_out_file.write(("Min Num: %s\n" % min_num).encode('utf-8'))
            feature_out_file.write(("Average Num: %s\n" % average_num).encode('utf-8'))
            feature_out_file.write(("Median Num: %s\n" % median_num).encode('utf-8'))
            feature_out_file.write(("Quartile Num: %s, %s\n" % (quartile1, quartile2)).encode('utf-8'))
            # 结果输出到console
            print("---------------------------------------------")
            print("Feature Name: %s" % feature_name)
            print("Missing Num: %s" % missing_num)
            print("Max Num: %s" % max_num)
            print("Min Num: %s" % min_num)
            print("Average Num: %s" % average_num)
            print("Median Num: %s" % median_num)
            print("Quartile Num: %s, %s" % (quartile1, quartile2))

    def impute_missing_values(self):
        """
        对dataset1中的["score_dif", "no_score_prob", "safety_prob", "WPA", "win_prob"]五个属性值进行填补
        并对填补后的数据分别画三种图展示结果
        """
        out_path = r'../results/imputed_figures'
        # 访问数据库取出5个属性对应的全部值
        sql = "SELECT %s,%s,%s,%s,%s FROM %s" % ("scorediff", "no_score_prob", "safety_prob", "WPA", "win_prob", dataset1_table_name)
        self.cursor.execute(sql)
        values = self.cursor.fetchall()
        processor = NumProcessor()
        # 分别用四种策略对缺失值进行填充
        for strategy in [1,2,3,4]:
            strategy_out_path = os.path.join(out_path, "strategy"+str(strategy))
            feature_values = processor.impute_missing_values(values, strategy)
            for i in range(len(impute_feature_list)):
                feature_name = impute_feature_list[i]
                value_set = feature_values[i]
                # 分别对该feature画三种图
                processor.draw_histogram(value_set, feature_name, strategy_out_path)
                processor.draw_qq_plot(value_set, feature_name, strategy_out_path)
                processor.draw_box(value_set, feature_name, strategy_out_path)


class FilenameError(Exception):
    pass


