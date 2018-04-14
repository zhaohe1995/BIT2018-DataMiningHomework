# -*- coding:UTF-8 -*-
"""
Some parameters and GLOBAL VARs
@author: ZhaoHe
"""

# database info
host = "127.0.0.1"
port = 3306
user = "root"
passwd = "1995328"
dbname = "datamininghomework"
dataset1_table_name = "play_by_play"
dataset2_table_name = "building_permits"

# 进行缺失值填补的几个属性列名
impute_feature_list = ["score_dif", "no_score_prob", "safety_prob", "WPA", "win_prob"]