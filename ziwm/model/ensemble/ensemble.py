#!/usr/bin/python2.7

from ziwm.model.classifier.base import Classifier


class Ensemble(Classifier):
    '''
    Base class for all ensembles
    '''
    
    @staticmethod
    def all_ensemble_types():
        from ziwm.model.ensemble.bagging.bagging import Bagging
        from ziwm.model.ensemble.random_subspace.random_subspace import RandomSubspace
        from ziwm.model.ensemble.randomnetworks.random_networks import RandomNetworks
        
        ensemble_types = []
        ensemble_types.append(Bagging)
        ensemble_types.append(RandomSubspace)
        ensemble_types.append(RandomNetworks)
        return ensemble_types
    
    #def __init__(self, base_classifiers, voting_system):
    #    '''
    #    Description...
    #    '''
    #    self.base_classifiers = base_classifiers
    #    self.voting_system = voting_system

    def __init__(self, voting_system, classifier_type, classifier_count):
        self.voting_system = voting_system
        classifiers = []
        for i in range(classifier_count):
            classifiers.append(classifier_type())
        self.base_classifiers = classifiers


