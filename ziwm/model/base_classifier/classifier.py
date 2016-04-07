#!/usr/bin/python2.7

from abc import ABCMeta, abstractmethod


class Classifier(object):
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
        from ziwm.model.base_classifier.mock_classifier.mock_classifier import MockClassifier
        from ziwm.model.base_classifier.extreme_learning_machine.extreme_learning_machine import ExtremeLearningMachine
        from ziwm.model.base_classifier.back_propagation_pybrain.back_propagation_pybrain import BackPropagationPyBrain

        models = []
        models.append(MockClassifier())
        models.append(ExtremeLearningMachine())
        models.append(BackPropagationPyBrain())
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
        Train model with a given mock_classifier
        X : 
            Feature matrix
        Y : 
            Label vector
        '''
        pass
