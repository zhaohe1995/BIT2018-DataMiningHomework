# -*- coding:UTF-8 -*-
"""
@author: ZhaoHe
"""

from src.data import Data
from src.config import dataset2_table_name

if __name__ == "__main__":
    data = Data()
    # 处理数据集的标称属性
    data.process_nom_features(data.dataset2_nom_feature_list, dataset2_table_name)