import numpy as np

from ziwm.model.base_classifier.baseclassifier import BaseClassifier
from ziwm.model.base_classifier.extreme_learning_machine.extreme_learning_machine import ExtremeLearningMachine


class RandomNetworks(BaseClassifier):
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
        return "Random Networks"

    def predict(self, x_test):
        '''
        Description...
        '''
        results = []
        for member in self.__committee:
            result = member.predict(x_test)
            results.append(result)
        return self.__majority_voting(results)

    def train(self, x, y):
        '''
        Mozna przyspieszyc wykonujac search_param jednokrotnie
        i trenujac bezposrednio metoda member.extreme_learning_machine.train
        '''
        for member in self.__committee:
            member.train(x, y)


    @staticmethod
    def __majority_voting(results):
        results_T = map(list, zip(*results))
        print 'rezulataty transponowane: ', results_T
        voting_result = []
        for row in results_T:
            rounded_row = [abs(int(round(n, 0))) for n in row]
            counts = np.bincount(rounded_row)
            voting_result.append(np.argmax(counts))
        print 'wyniki glosowania: ', voting_result





