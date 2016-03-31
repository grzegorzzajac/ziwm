from abc import ABCMeta, abstractmethod

class Model(object):
    '''
    Base class for predictive models
    '''
    __metaclass__ = ABCMeta

    @staticmethod
    def all_models():
        '''
        Returns objects of all available predictive model classes
        (should be subclasses of Model class)
        '''
        from ziwm.model.mock.mock import MockClassifier

        models = []
        models.append(MockClassifier())
        return models

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
        Train model with a given mock
        X : 
            Feature matrix
        Y : 
            Label vector
        '''
        pass
