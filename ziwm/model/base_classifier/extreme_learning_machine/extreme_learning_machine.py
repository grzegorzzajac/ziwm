import sys
from os import path

import elm
import numpy as np

from ziwm.model.base_classifier.classifier import Classifier

sys.path.append(path.abspath('..'))
sys.path.append(path.abspath('./../../..'))


class ExtremeLearningMachine(Classifier):
    '''
    Description...
    '''
    def __init__(self):
        self.__elmk = elm.ELMKernel()

    @staticmethod
    def name():
        return "Extreme Learning Machine"

    def predict(self, x_test):
        '''
        Description...
        '''
        zero_x_test = np.hstack((np.zeros((x_test.shape[0], 1)), x_test))
        result = self.__elmk.test(zero_x_test).predicted_targets
        #return abs(result.round(0))
        return result

    def train(self, x, y):
        '''
        Description...
        '''
        y_T = np.array([y]).T
        data = np.concatenate((y_T, x), axis=1)
        self.__elmk.search_param(data, cv="kfold", of="accuracy", eval=10)
        return self.__elmk.train(data)


