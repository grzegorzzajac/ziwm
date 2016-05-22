#!/usr/bin/python2.7

import numpy as np
from ziwm.model.ensemble.ensemble import Ensemble


class Bagging(Ensemble):
    def __init__(self, voting_system, classifier_type, classifier_count):
        super(Bagging, self).__init__(voting_system, classifier_type, classifier_count)

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
        return self.voting_system.vote(results)
    
    def train(self, x, y, class_number=-1):
        '''
        Mozna przyspieszyc wykonujac search_param jednokrotnie
        i trenujac bezposrednio metoda member.extreme_learning_machine.train
        '''
        for member in self.base_classifiers:
            new_x, new_y = self.__create_individual_dataset(x, y)
            member.train(new_x, new_y, class_number)

    @staticmethod
    def __create_individual_dataset(X, Y):
        row_count = X.shape[0]
        
        random_indices = np.random.randint(0, high=row_count, size=row_count)

        X_random = X[random_indices, :]
        Y_random = Y[random_indices]
        return X_random, Y_random




