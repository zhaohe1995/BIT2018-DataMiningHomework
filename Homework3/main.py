# -*- coding:UTF-8 -*-
"""
Train models and show the results
@author: ZhaoHe
"""
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.data import Data
from src.config import train_path, test_path, submission_sample, submission_dir, images_dir
from src.classifiers import classifier_xgboost, classifier_dicisionTree, classifier_SVM
from src.clusters import cluster_KMeans, cluster_Hierarchical, cluster_Spectral

def get_submission(pred, model_name):
    submission_df = pd.read_csv(submission_sample, header=0)
    for i in range(len(pred)):
        submission_df.ix[i,'Survived'] = pred[i]
    out_file = os.path.join(submission_dir, model_name)
    submission_df.to_csv(out_file, index=False)

def pca(X_multi_dim):
    pca_model = PCA(n_components=2)
    X_2_dim = pca_model.fit_transform(X_multi_dim)
    return X_2_dim

def plot_result(pca_X, label_Y, title):
    class0_x, class0_y = [], []
    class1_x, class1_y = [], []
    for i in range(len(pca_X)):
        if label_Y[i] == 0:
            class0_x.append(pca_X[i][0])
            class0_y.append(pca_X[i][1])
        elif label_Y[i] == 1:
            class1_x.append(pca_X[i][0])
            class1_y.append(pca_X[i][1])
    plt.plot(class0_x, class0_y, 'or', label='No-Survived')
    plt.plot(class1_x, class1_y, 'ob', label='Survived')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(images_dir, title))
    plt.close()


def train_models(train_X, train_Y, test_X):
    # Classify using Xgboost
    pred_Y = classifier_xgboost(train_X, train_Y, test_X)
    get_submission(pred_Y, 'xgboost')

    # Classify using decisionTree
    pred_Y = classifier_dicisionTree(train_X, train_Y, test_X)
    get_submission(pred_Y, 'decisionTree')

    # Classify using SVM
    pred_Y = classifier_SVM(train_X, train_Y, test_X)
    get_submission(pred_Y, 'SVM')

    # Cluster using KMeans
    pred_Y = cluster_KMeans(train_X, test_X)
    get_submission(pred_Y, 'KMeans')

    # Cluster using Herichical
    pred_Y = cluster_Hierarchical(test_X)
    get_submission(pred_Y, 'Hierachical')

    # Cluster using Spectral
    pred_Y = cluster_Spectral(test_X)
    get_submission(pred_Y, 'Spectral')

def visualize_models(train_X, train_Y, test_X):
    # PCA on X
    new_train_X = pca(train_X)
    new_test_X = pca(test_X)

    # Visualize the train set
    plot_result(new_train_X, train_Y, 'Train Set')

    # Visualize the Xgboost
    sub_file = os.path.join(submission_dir, 'Xgboost')
    data = pd.read_csv(sub_file, header=0)
    label_y = data['Survived']
    plot_result(new_test_X, label_y, 'Xgboost')

    # Visualize the decisionTree
    sub_file = os.path.join(submission_dir, 'decisionTree')
    data = pd.read_csv(sub_file, header=0)
    label_y = data['Survived']
    plot_result(new_test_X, label_y, 'DecisionTree')

    # Visualize the SVM
    sub_file = os.path.join(submission_dir, 'SVM')
    data = pd.read_csv(sub_file, header=0)
    label_y = data['Survived']
    plot_result(new_test_X, label_y, 'SVM')

    # Visualize the KMeans
    sub_file = os.path.join(submission_dir, 'KMeans')
    data = pd.read_csv(sub_file, header=0)
    label_y = data['Survived']
    plot_result(new_test_X, label_y, 'KMeans')

    # Visualize the Hierachical
    sub_file = os.path.join(submission_dir, 'Hierachical')
    data = pd.read_csv(sub_file, header=0)
    label_y = data['Survived']
    plot_result(new_test_X, label_y, 'Hierahical')

    # Visualize the Spectral
    sub_file = os.path.join(submission_dir, 'Spectral')
    data = pd.read_csv(sub_file, header=0)
    label_y = data['Survived']
    plot_result(new_test_X, label_y, 'Spectral')




if __name__ == "__main__":
    data = Data()
    train_X, train_Y = data.load_data(train_path)
    test_X = data.load_data(test_path, train = False)

    #train_models(train_X, train_Y, test_X)
    visualize_models(train_X, train_Y, test_X)