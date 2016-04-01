import numpy as np
from ziwm.model.ensemble.ensemble import Ensemble


class RandomNetworks(Ensemble):
    '''
    Description...
    '''

    def __init__(self, base_classifiers, voting_system):
        super(RandomNetworks, self).__init__(base_classifiers, voting_system)

    @staticmethod
    def name():
        return "Random Networks"

    def predict(self, x_test):
        '''
        Description...
        '''
        results = []
        for member in self.base_classifiers:
            result = member.predict(x_test)
            results.append(result)
        print 'partial results:\n', results
        return self.voting_system.vote(results)

    def train(self, x, y):
        '''
        Mozna przyspieszyc wykonujac search_param jednokrotnie
        i trenujac bezposrednio metoda member.extreme_learning_machine.train
        '''
        for member in self.base_classifiers:
            member.train(x, y)






