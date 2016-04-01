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





