#!/usr/bin/python2.7

import random
import numpy as np
from ziwm.model.ensemble.ensemble import Ensemble


class Bagging(Ensemble):
    '''
    Bagging description...
    '''

    def __init__(self, base_classifiers, voting_system):
        super(Bagging, self).__init__(base_classifiers, voting_system)

    @staticmethod
    def name():
        return "Bagging"

    def predict(self, x_test):
        '''
        Description...
        '''
        results = []
        for member in self.base_classifiers:
            result = member.predict(x_test)
            results.append(result)
        #print 'partial results:\n', results
        return self.voting_system.vote(results)

    def train(self, x, y):
        '''
        Mozna przyspieszyc wykonujac search_param jednokrotnie
        i trenujac bezposrednio metoda member.extreme_learning_machine.train
        '''
        for member in self.base_classifiers:
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




