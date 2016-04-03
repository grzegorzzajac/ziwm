#!/usr/bin/python2.7

from abc import abstractmethod
from ziwm.model.base_classifier.classifier import Classifier


class Ensemble(Classifier):
    '''
    Base class for all ensembles
    '''

    def __init__(self, base_classifiers, voting_system):
        '''
        Description...
        '''
        self.base_classifiers = base_classifiers
        self.voting_system = voting_system





