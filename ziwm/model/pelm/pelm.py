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
    __elmk = elm.ELMKernel()

    def name(self):
        return "Extreme Learning Machine"

    def predict(self, x):
        '''
        Description...
        '''
        return self.__elmk.test(x).get_accuracy()

    def train(self, x, y):
        '''
        Description...
        '''
        y_T = np.array([y]).T
        data = np.concatenate((y_T, x), axis=1)
        self.__elmk.search_param(data, cv="kfold", of="accuracy", eval=10)
        return self.__elmk.train(data).get_accuracy()


