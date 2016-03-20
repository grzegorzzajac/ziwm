from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    '''
    Base class for predictive models
    '''

    @staticmethod
    def all_models():
        '''
        Returns objects of all available predictive model classes
        (should be subclasses of Model class)
        '''
        from ziwm.model.classifier.mock import MockClassifier

        models = []
        models.append(MockClassifier())
        return models
    
    @abstractmethod
    def name(self):
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
        Train model with a given dataset
        X : 
            Feature matrix
        Y : 
            Label vector
        '''
        pass
