# -*- coding:UTF-8 -*-
"""
@author: ZhaoHe
"""

import os, math
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats
import numpy as np
from fancyimpute import SimpleFill, KNN, MICE

class NumProcessor(object):
    def __init__(self, feature_name=None, value_set=None):
        self.feature_name = feature_name
        self.value_set = value_set   # 一个存放了全部属性值的list
        if value_set:
            self.value_num = len(value_set)  # 数据属性值的总个数
            self.new_value_set = self.drop_missing_values(self.value_set)
        else:
            self.new_value_set = None
            self.value_num = 0  # 数据属性值的总个数

    def pre_process(self):
        new_value_set = self.drop_missing_values(self.value_set)
        missing_num = self.get_missing_num(new_value_set)
        max_num = self.get_maximum(new_value_set)
        min_num = self.get_minimum(new_value_set)
        average_num = self.get_average(new_value_set)
        median_num = self.get_median(new_value_set)
        quartile1, quartile2 = self.get_quartile(new_value_set)
        return (missing_num, max_num, min_num, average_num, median_num, quartile1, quartile2)

    def drop_missing_values(self, value_set):
        """
        除去数据集中的空值（包括NA、Null）
        将剩余值转化为浮点数
        :param value_set: 原始数据列表
        :return: 去除了缺失值的数据列表
        """
        new_value_set = []
        for value in value_set:
            if value and value != 'NA':
                new_value_set.append(float(value))
        return new_value_set

    def get_missing_num(self, value_set):
        return self.value_num - len(value_set)

    def get_maximum(self, value_set):
        return max(value_set)

    def get_minimum(self, value_set):
        return min(value_set)

    def get_average(self, value_set):
        value_num = len(value_set)
        return sum(value_set)/value_num

    def get_median(self, value_set):
        value_set = sorted(value_set)
        value_num = len(value_set)
        if value_num % 2 == 0:
            median = (value_set[value_num//2] + value_set[value_num//2-1])/2
        else:
            median = value_set[(value_num-1)//2]
        return median

    def get_quartile(self, value_set):
        # 求下分位数Q1、上分位数Q2
        # 首先分别确定Q1、Q2位置
        value_set = sorted(value_set)
        value_num = len(value_set)
        Q1_index = (value_num + 1) * 0.25
        Q2_index = (value_num + 1) * 0.75
        # 四分位数的值在index向下取整位置与向上取整位置之间
        Q1_value1 = value_set[int(math.floor(Q1_index))]
        Q1_value2 = value_set[int(math.ceil(Q1_index))]
        percent = Q1_index - math.floor(Q1_index)
        Q1_value = percent * Q1_value1 + (1-percent) * Q1_value2

        Q2_value1 = value_set[int(math.floor(Q2_index))]
        Q2_value2 = value_set[int(math.ceil(Q2_index))]
        percent = Q2_index - math.floor(Q2_index)
        Q2_value = percent * Q2_value1 + (1 - percent) * Q2_value2

        return(Q1_value, Q2_value)

    def draw_histogram(self, value_set, feature_name, out_path):
        out_path = os.path.join(out_path, "histogram", feature_name)

        # 统计value_set中共有多少个不同的值
        '''
        value_dict = dict()
        for value in value_set:
            if value not in value_dict:
                value_dict[value] = 1
            else:
                value_dict[value] += 1
        dif_value_num = len(value_dict)'''
        # 条形图的区间数为 不同值个数除以20取整
        # 画图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.grid(True)
        ax.hist(value_set, bins=100, color='g')
        plt.title(feature_name)
        plt.xlabel("Value")
        plt.ylabel("Freq")
        #plt.show()
        plt.savefig(out_path)
        plt.close()

    def draw_qq_plot(self, value_set, feature_name, out_path):
        out_path = os.path.join(out_path, "q-q", feature_name)

        stats.probplot(value_set, dist="norm", plot=pylab)
        pylab.title(feature_name)
        pylab.savefig(out_path)
        pylab.close()

    def draw_box(self, value_set, feature_name, out_path):
        out_path = os.path.join(out_path, "box", feature_name)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(value_set, sym="o", whis=1.5, labels=[feature_name])
        plt.savefig(out_path)
        plt.close()

    def impute_missing_values(self, value_set, strategy):
        """
        对原始数据矩阵进行填充
        :param value_set: 待处理的原始数据矩阵
        :param strategy: 1：剔除缺失值 2：高频值填充 3：属性相关关系填充 4：数据对象相似性填充
        :return: 进行填充过的数据矩阵，类型为list: (col1, col2, ...)
        """
        # 以剔除缺失值的方法进行处理
        if strategy == 1:
            new_value_set = []
            for data_sample in value_set:
                new_data_sample = []
                if None in data_sample or 'NA' in data_sample:
                    continue
                else:
                    for data in data_sample:
                        new_data_sample.append(float(data))
                new_value_set.append(new_data_sample)
            value_array = np.array(new_value_set)

        elif strategy in [2,3,4]:
            # 将value_set矩阵转化为numpy矩阵，并将其中的缺失值用np.nan替换
            new_value_set = []
            for data_sample in value_set:
                new_data_sample = []
                for data in data_sample:
                    if data and data != 'NA':
                        new_data_sample.append(float(data))
                    else:
                        new_data_sample.append(np.nan)
                new_value_set.append(new_data_sample)
            value_array = np.array(new_value_set)

            # 以最高频值进行填补，由于均为概率类的数值属性，所以用平均数代替
            if strategy == 2:
                value_array = SimpleFill(fill_method = "mean").complete(value_array)

            # 以属性相关关系进行填补，取相关性最高的三个属性做
            elif strategy == 3:
                value_array = MICE(n_nearest_columns = 3).complete(value_array)

            # 以数据对象相似性进行填补，取相似度最高的10个数据对象
            elif strategy == 4:
                for batch in range(len(value_array)//1000+ 1):
                    value_array[batch*1000 : min(batch*1000+1000, len(value_array))] = \
                        KNN(k = 10).complete(value_array[batch*1000 : min(batch*1000+1000, len(value_array))])
        else:
            raise ArgInputError("The strategy should be in (1,2,3,4)!")

        # 将填充过的数据矩阵按feature_col转换为n个col的list
        feature_col_list = []
        for i in range(len(value_array[0])):
            feature_col_list.append(value_array[:,i].tolist())
        return feature_col_list

class ArgInputError(Exception):
    pass










