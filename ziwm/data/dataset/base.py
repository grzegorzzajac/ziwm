from abc import ABCMeta, abstractmethod

class Dataset(object):
    '''
    Base class for a abstraction over dataset.
    It should be used for loading dataset from files and extracting features.
    '''
    __metaclass__=ABCMeta   

    @staticmethod
    def all_datasets():
        '''
        Returns objects of all available dataset classes
        (should be subclasses of Dataset class).
        '''
        from ziwm.data.dataset.mock import MockDataset

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
        Arbitrary name of a given dataset
        '''
        pass
    
    @abstractmethod
    def load(self):
        '''
        Load feature matrix and labels vector from a dataset
        '''
        pass
