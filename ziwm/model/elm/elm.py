#!/usr/bin/python2

import sys
from os import path

import numpy as np

import elm
from ziwm.model.base import Model

sys.path.append(path.abspath('..'))
sys.path.append(path.abspath('./../../..'))


class ExtremeLearningMachine(Model):
    '''
    Description...
    '''

    def name(self):
        return "Extreme Learning Machine"

    def predict(self, X):
        '''
        Description...
        '''
        row_mins = X.min(axis=1)
        row_maxs = X.max(axis=1)
        Y = ((row_mins == 1) & (row_maxs == 1))
        Y = Y * 1.0
        return Y

    def train(self, X, Y):
        '''
        Description...
        '''

        # create a classifier
        elmk = elm.ELMKernel()

        # search for best parameter for this dataset
        # define "kfold" cross-validation method, "accuracy" as a objective function
        # to be optimized and perform 10 searching steps.
        # best parameters will be saved inside 'elmk' object
        #elmk.search_param(data, cv="kfold", of="accuracy", eval=10)

        #train and test
        # results are Error objects
        #tr_result = elmk.train(tr_set)
        print X
        print Y
        pass

if __name__ == "__main__":

    X = np.arange(50).reshape(10,5) / 50.0
    X = np.vstack((X, np.ones(5)))
    X = np.vstack((np.ones(5), X))
    print("X:\n{}".format(X))

    model = ExtremeLearningMachine()
    model.train(X, None)
    Y_predicted = model.predict(X)

    print("predicted Y:\n{}".format(Y_predicted))

    print("Done")

