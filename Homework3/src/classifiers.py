# -*- coding:UTF-8 -*-
"""
3 Classifers
@author: ZhaoHe
"""
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


def classifier_xgboost(train_X, train_Y, test_X):
    gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    gbm.fit(train_X, train_Y)
    predictions = gbm.predict(test_X)
    return predictions

def classifier_dicisionTree(train_X, train_Y, test_X):
    dtc = DecisionTreeClassifier(random_state=0, criterion='gini', max_leaf_nodes=10)
    dtc.fit(train_X, train_Y)
    predictions = dtc.predict(test_X)
    return predictions

def classifier_SVM(train_X, train_Y, test_X):
    svc = svm.SVC()
    svc.fit(train_X, train_Y)
    predictions = svc.predict(test_X)
    return predictions
