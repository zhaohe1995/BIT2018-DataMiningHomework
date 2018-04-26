# -*- coding:UTF-8 -*-
"""
Load data
@author: ZhaoHe
"""
from .config import train_path, test_path
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class Data(object):
    def __init__(self):
        pass

    def load_data(self, path, train=True):
        # 读取数据
        data = pd.read_csv(path, header = 0)

        # 选定将使用的特征
        feature_columns_to_use = ['Pclass','Sex','Age','Fare','Embarked','SibSp','Parch']
        nonnumric_columns = ['Sex','Embarked']

        # 选取特征对应的数据，并对数据进行缺失值填补
        X = data[feature_columns_to_use]
        X_imputed = DataFrameImputer().fit_transform(X)

        # 进行特征工程，构造两个新特征
        X_imputed['Family'] = data['SibSp'] + data['Parch']
        X_imputed['IsAlone'] = pd.Series([1 if X_imputed['Family'][i]>0 else 0 for i in range(len(X_imputed))])

        # 将非数值属性转化为数值属性
        le = LabelEncoder()
        for feature in nonnumric_columns:
            X_imputed[feature] = le.fit_transform(X_imputed[feature])

        # 准备用于输入到模型中的数据
        input_X = X_imputed.as_matrix()
        if train == False:
            return input_X
        else:
            input_Y = data['Survived']
            return input_X, input_Y

class DataFrameImputer(TransformerMixin):
    """We'll impute missing values using the median for numeric columns and the most
        common value for string columns.
        This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948"""
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')
                               else X[c].median() for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

