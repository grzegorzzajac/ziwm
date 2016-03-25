#!/usr/bin/python2

from os import path
import sys
from ziwm.model.base import Model
import numpy as np


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

