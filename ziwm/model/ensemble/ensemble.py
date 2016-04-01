from ziwm.model.base_classifier.baseclassifier import BaseClassifier
from abc import abstractmethod


class Ensemble(BaseClassifier):
    '''
    Base class for all ensembles
    '''

    def __init__(self, base_classifiers, voting_system):
        '''
        Description...
        '''
        self.base_classifiers = base_classifiers
        self.voting_system = voting_system

    @staticmethod
    @abstractmethod
    def name():
        '''
        Arbitrary name of a given model
        '''
        pass

    @abstractmethod
    def predict(self, X):
        '''
        Make a prediction of a X feature matrix
        '''
        pass

    @abstractmethod
    def train(self, X, Y):
        '''
        Train model
        X :
            Feature matrix
        Y :
            Label vector
        '''
        pass




