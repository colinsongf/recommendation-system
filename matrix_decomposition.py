# -*- coding: utf-8 -*-
""""
Recommendation using Matrix Decomposition
"""
from __future__ import print_function, division, absolute_import

import os
import numpy as np
np.random.seed(0)  # for reproducibility
from numpy.linalg import norm
import matplotlib.pyplot as plt
import six.moves.cPickle as cPickle
from data_process import load_netflix_pkl


def plot_md_result(pkl_path):
    """plot results of gradient descend"""
    results = cPickle.load(open(pkl_path, 'rb'))
    # plot J
    plt.figure()
    plt.plot(results['epochs'], results['Js'])
    plt.xlabel('epoch')
    plt.ylabel('J')
    plt.title('J (k=%d, lamb=%f, alpha=%f)' % (results['k'], results['lamb'], results['alpha']))
    plt.grid(True)
    plt.show()

    # plot RMSE
    plt.figure()
    plt.plot(results['epochs'], results['rmses'])
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE (k=%d, lamb=%f, alpha=%f)' % (results['k'], results['lamb'], results['alpha']))
    plt.grid(True)
    plt.show()


def J_func(A, X, U, V, lamb):
    """objective function of matrix-decomposition algorithm"""
    return 0.5 * norm(A*(X-U.dot(V.T)))**2 + lamb * norm(U)**2 + lamb * norm(V)**2


def train_gd(X_train, X_test, k=50, lamb=0.01, alpha=0.0001, max_epoch=150, verbose=True, save_path=None):
    """train the model using gradient descend"""
    eps = 1000
    U = np.random.uniform(-0.1, 0.1, size=(10000, k))
    V = np.random.uniform(-0.1, 0.1, size=(10000, k))
    A, B = (X_train != 0), (X_test != 0)  # indicator matrix
    B_sum = np.sum(B)

    epochs, Js, rmses = [], [], []
    epoch = 0
    # too_slow = [False] * 10  # for early-stopping
    while epoch <= max_epoch-1:
        dJ_U = (A*(U.dot(V.T)-X_train)).dot(V) + 2*lamb*U
        dJ_V = (A*(U.dot(V.T)-X_train)).T.dot(U) + 2*lamb*V

        update_rate = np.mean([norm(alpha*dJ_U)/norm(U), norm(alpha*dJ_V)/norm(V)])
        U += -alpha * dJ_U
        V += -alpha * dJ_V
        # too_slow[epoch % 10] = (True if update_rate < 0.0001 else False)

        X_predict = U.dot(V.T) * B
        rmse = np.sqrt(np.sum((X_predict-X_test)**2) / B_sum)  # rmse on test data
        J = J_func(A, X_train, U, V, lamb)  # calculate new cost
        epoch += 1
        epochs.append(epoch)
        Js.append(J)
        rmses.append(rmse)
        if verbose:
            print(epochs[-1], Js[-1], rmses[-1], 'update rate:', update_rate)
        # if all(too_slow): break
    result = {
        'k': k,
        'lamb': lamb,
        'alpha': alpha,
        'epochs': epochs,
        'Js': Js,
        'rmses': rmses
    }
    if save_path != None:
        cPickle.dump(result, open(save_path, 'wb'))
    return result


if __name__ == '__main__':
    if not os.path.exists('result.pkl'):
        X_train, X_test = load_netflix_pkl('netflix.pkl')
        train_gd(X_train, X_test, save_path='result.pkl')
    plot_md_result('result.pkl')
