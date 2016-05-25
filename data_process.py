# -*- coding: utf-8 -*-
"""
Data Pre-processing
"""
from __future__ import print_function, division, absolute_import

import os
import numpy as np
import six.moves.cPickle as cPickle
from tqdm import tqdm


def load_txt(save_path='netflix.pkl'):
    """load data from original txt, save as pickle format"""
    users = np.loadtxt('users.txt')

    X_train = np.zeros((10000, 10000), dtype=np.int16)
    print('... loading train data')
    for line in tqdm(open('netflix_train.txt')):
        user_id, movie_id, score, date = line.split()
        user_id, movie_id, score = int(user_id), int(movie_id), int(score)
        X_train[users == user_id, movie_id-1] = score

    X_test = np.zeros((10000, 10000), dtype=np.int16)
    print('... loading test data')
    for line in tqdm(open('netflix_test.txt')):
        user_id, movie_id, score, date = line.split()
        user_id, movie_id, score = int(user_id), int(movie_id), int(score)
        X_test[users == user_id, movie_id-1] = score

    netflix = (X_train, X_test)
    cPickle.dump(netflix, open(save_path, 'wb'))


def load_netflix_pkl(filename='netflix.pkl'):
    """load user behavior matrix from pickle format (returned by load_txt)"""
    if not os.path.exists(filename):
        load_txt()
    X_train, X_test = cPickle.load(open(filename, 'rb'))
    X_train = np.asarray(X_train, dtype=np.float)
    X_test = np.asarray(X_train, dtype=np.float)
    return X_train, X_test


if __name__ == '__main__':
    load_txt()
