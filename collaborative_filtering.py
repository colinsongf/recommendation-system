# -*- coding: utf-8 -*-
""""
Recommendation using Collaborative Filtering
"""
from __future__ import print_function, division, absolute_import

import numpy as np
from data_process import load_netflix_pkl


def cosine_similarity(X):
    """Calc Similarity Matrix using Cosine Measure"""
    user_num = X.shape[0]
    X_norm = X_train / np.linalg.norm(X, axis=1).reshape((user_num, 1))  # normalization along each row
    return np.dot(X_norm, X_norm.T)


@profile
def recommendation_cf(X_train, X_test, similarity_func=cosine_similarity):
    """Collaborative Filtering"""
    print('... training')
    similarity = similarity_func(X_train)  # calc similarity matrix
    X_predict = np.dot(similarity, X_train) / np.dot(similarity, X_train!=0)  # predicted

    print('... testing')
    X_predict *= (X_test != 0)  # mask unknown elements
    rmse = np.sqrt(np.sum((X_predict-X_test)**2) / np.sum(X_test!=0))
    return rmse


if __name__ == '__main__':
    print('... loading data from pkl')
    X_train, X_test = load_netflix_pkl('netflix.pkl')
    rmse = recommendation_cf(X_train, X_test)
    print('rmse:', rmse)
