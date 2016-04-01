import random
import numpy as np
from ziwm.model.base_classifier.baseclassifier import BaseClassifier
from ziwm.model.base_classifier.extreme_learning_machine.extreme_learning_machine import ExtremeLearningMachine


class Bagging(BaseClassifier):
    '''
    Description...
    '''

    def __init__(self, committee_size=3):
        '''
        Description...
        '''
        self.__committee = []
        if committee_size < 2:
            committee_size = 2
        for member in range(committee_size):
            member = ExtremeLearningMachine()
            self.__committee.append(member)

    @staticmethod
    def name():
        return "Bagging"

    def predict(self, x_test):
        '''
        Description...
        '''
        results = []
        for member in self.__committee:
            result = member.predict(x_test)
            results.append(result)
        return np.average(results, axis=0)
        #return int(np.average(results, axis=0).round(0))

    def train(self, x, y):
        '''
        Description...
        '''
        # mozna przyspieszyc wykonujac search_param jednokrotnie i trenujac bezposrednio metoda member.extreme_learning_machine.train
        for member in self.__committee:
            new_x, new_y = self.__create_individual_dataset(x, y)
            member.train(new_x, new_y)

    @staticmethod
    def __create_individual_dataset(x, y):
        size = x.shape[0]
        new_x = np.zeros(x.shape)
        new_y = np.zeros(y.shape)
        for row in range(0, size):
            random_row = random.randint(0, size-1)
            new_x[row, :] = x[random_row, :]
            new_y[row] = y[random_row]
            #print x[row, :], y[row], '->', new_x[row, :], new_y[row]
        #print x, y
        #print '<<<<<<<<<<<<<<>>>>>>>>>>>>>>'
        #print new_x, new_y
        return new_x, new_y




