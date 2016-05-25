# -*- coding: utf-8 -*-
"""
Study of hyper-parameters in Matrix Decomposition method
"""
from __future__ import print_function, division, absolute_import

import os
import numpy as np
import matplotlib.pyplot as plt
from data_process import load_netflix_pkl
from matrix_decomposition import train_gd
import six.moves.cPickle as cPickle


def param_search():
    X_train, X_test = load_netflix_pkl('netflix.pkl')

    result_list = []
    for k in (20, 50, 100):
        for lamb in (0.001, 0.01, 0.1):
            print('\n... running for: k = %d, lamb = %f' % (k, lamb))
            result = train_gd(X_train, X_test, k=k, lamb=lamb)
            result_list.append(result)
    cPickle.dump(result_list, open('results_list.pkl', 'wb'))
    return result_list


def plot_param_result(data_path='result_list.pkl'):
    result_list = cPickle.load(open(data_path, 'rb'))
    print('k    lamb    min-rmse')
    for result in result_list:
        print(result['k'], result['lamb'], np.min(result['rmses']))
    epochs = result_list[0]['epochs']
    # plot 3 curves in one figure
    def _plot_one_fig(plt, x, y_list, xlabel, ylabel, legend):
        plt.figure()
        plt.plot(x, y_list[0])
        plt.plot(x, y_list[1])
        plt.plot(x, y_list[2])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend(legend)
        plt.show()

    # k = 50, lamb change
    Js_list = [result_list[3]['Js'], result_list[4]['Js'], result_list[5]['Js']]
    rmses_list = [result_list[3]['rmses'], result_list[4]['rmses'], result_list[5]['rmses']]
    legend = ('lamb=0.001', 'lamb=0.01', 'lamb=0.1')
    _plot_one_fig(plt, epochs, Js_list, xlabel='epochs', ylabel='J', legend=legend)
    _plot_one_fig(plt, epochs, rmses_list, xlabel='epochs', ylabel='RMSE', legend=legend)

    # lamb = 0.01, k change
    Js_list = [result_list[1]['Js'], result_list[4]['Js'], result_list[7]['Js']]
    rmses_list = [result_list[1]['rmses'], result_list[4]['rmses'], result_list[7]['rmses']]
    legend = ('k=20', 'k=50', 'k=100')
    _plot_one_fig(plt, epochs, Js_list, xlabel='epochs', ylabel='J', legend=legend)
    _plot_one_fig(plt, epochs, rmses_list, xlabel='epochs', ylabel='RMSE', legend=legend)
    pass


if __name__ == '__main__':
    if not os.path.exists('result_list.pkl'):
        param_search()
    plot_param_result(data_path='result_list.pkl')
