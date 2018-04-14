# -*- coding:UTF-8 -*-
"""
@author: ZhaoHe
"""
from src.data import Data
from src.config import dataset1_table_name, dataset2_table_name


if __name__ == "__main__":
    data = Data()
    # 分别处理两个数据集的标称属性
    data.process_nom_features(data.dataset1_nom_feature_list, dataset1_table_name)
    data.process_nom_features(data.dataset2_nom_feature_list, dataset2_table_name)

    # 分别处理两个数据集的数值属性
    data.process_num_features(data.dataset1_num_feature_list, dataset1_table_name)
    data.process_num_features(data.dataset2_num_feature_list, dataset2_table_name)

    # 选取dataset1中的五个属性进行填充
    data.impute_missing_values()