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

        datasets = []
        datasets.append(MockDataset())
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
