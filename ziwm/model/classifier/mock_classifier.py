#!/usr/bin/python2.7

import numpy as np

from ziwm.model.classifier.base import Classifier


class MockClassifier(Classifier):
    '''
    A simple, mock_classifier mock_classifier. It does not learn.
    Always returns [0,1,0,...] if all features are equal to '1'
    and [1,0,0,...] otherwise
    '''
    def __init__(self):
        self.__class_num = 0

    def name(self):
        return "MockClassifier"
    
    def predict(self, X):
        '''
        Predicts [0,1,0,...] if all features are equal to '1',
        [1,0,0,...] otherwise
        '''
        row_mins = X.min(axis=1)
        row_maxs = X.max(axis=1)
        Y = ((row_mins == 1) & (row_maxs == 1))
        # FIXME: make it pretty
        first = [1] + [0] * (self.__class_num - 1)
        second = [0] + [1] + [0] * (self.__class_num - 2)
        Y = [second if y else first for y in Y]
        return Y
    
    def train(self, X, Y, class_number=-1):
        '''
        Save a number of classes
        '''
        self.__class_num = max(np.unique(Y).size, class_number)
    
if __name__ == "__main__":

    X = np.arange(50).reshape(10,5) / 50.0
    X = np.vstack((X, np.ones(5)))
    X = np.vstack((np.ones(5), X))
    print("X:\n{}".format(X))

    model = MockClassifier()
    model.train(X, None)
    Y_predicted = model.predict(X)
    
    print("predicted Y:\n{}".format(Y_predicted))

    print("Done")
    
