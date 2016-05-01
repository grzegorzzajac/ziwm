#!/usr/bin/python2.7

import neurolab as nl
import numpy as np

from ziwm.model.classifier.base import Classifier


class BackPropagationNeuroLab(Classifier):
    '''
    Description...
    '''
    def __init__(self):
        self.__class_zero_indexing = True

    @staticmethod
    def name():
        return "Back Propagation NeuroLab"

    def predict(self, x_test):
        y = []
        y_one_hot = self.__nn.sim(x_test)
        for one_hot_vector in y_one_hot:
            class_number = np.argmax(one_hot_vector)
            if not self.__class_zero_indexing:
                class_number += 1
            y.append(class_number)

        return np.array(y)

    def train(self, x, y):
        in_num = x.shape[1]  # input vector size
        out_num = np.unique(y).size  # number of classes
        hidden_num = (in_num + out_num) / 2  # number of neurons in hidden layer

        min_max = np.array([x.min(axis=0), x.max(axis=0)]).T  # min and max for each input

        '''
        The nuerolab implementation normalizes the output so we have to
        translate target vector from integers to one-hot, e.g.:
        [1]    [0, 1, 0]
        [2]    [0, 0, 1]
        [2] => [0, 0, 1]
        [0]    [1, 0, 0]
        [1]    [0, 1, 0]
        '''

        self.__class_zero_indexing = min(y) == 0  # indexing from 0 or 1?
        y_classes = np.zeros((y.size, out_num))  # one-hot target vector
        for i in range(y.size):
            if self.__class_zero_indexing:
                y_classes[i][y[i]] = 1
            else:
                y_classes[i][y[i] - 1] = 1

        self.__nn = nl.net.newff(min_max, [in_num, hidden_num, out_num])
        self.__nn.train(x, y_classes)
        #nl.train.train_gd(self.__nn, x, y_classes, lr=0.005, adapt=True, epochs=1000)
