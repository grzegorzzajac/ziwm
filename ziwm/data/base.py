#!/usr/bin/python2.7

from abc import ABCMeta, abstractmethod

class Dataset(object):
    '''
    Base class for a abstraction over mock_classifier.
    It should be used for loading mock_classifier from files and extracting features.
    '''
    __metaclass__=ABCMeta   

    @staticmethod
    def all_datasets():
        '''
        Returns objects of all available mock_classifier classes
        (should be subclasses of Dataset class).
        '''
        from ziwm.data.mock.mock import MockDataset
        from ziwm.data.iris.iris import IrisDataset
        from ziwm.data.breast_cancer.breast_cancer import BreastCancer
        from ziwm.data.dermatology.dermatology import Dermatology
        from ziwm.data.house_votes_84.house_votes_84 import HouseVotes84
        from ziwm.data.soybean_large.soybean_large import SoybeanLarge
        from ziwm.data.soybean_small.soybean_small import SoybeanSmall
        from ziwm.data.tic_tac_toe.tic_tac_toe import TickTackToeDataset
        from ziwm.data.wine.wine import Wine

        datasets = []
        datasets.append(IrisDataset())
        datasets.append(MockDataset())
        datasets.append(BreastCancer())
        #datasets.append(Dermatology())
        datasets.append(HouseVotes84())
        datasets.append(SoybeanLarge())
        #datasets.append(SoybeanSmall())
        datasets.append(TickTackToeDataset())
        #datasets.append(Wine())
        return datasets
    
    @abstractmethod
    def problem_type(self):
        '''
        Type of the problem:
        'regression' or 'classification'
        '''
        pass
    
    @abstractmethod
    def name(self):
        '''
        Arbitrary name of a given mock_classifier
        '''
        pass
    
    @abstractmethod
    def load(self):
        '''
        Load feature matrix and labels vector from a mock_classifier
        '''
        pass
