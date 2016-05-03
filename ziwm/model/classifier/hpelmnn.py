from ziwm.model.classifier.base import Classifier

from hpelm import ELM
import numpy as np

class HPELMNN(Classifier):
    
    def __init__(self):
        self.__hpelm = None
    
    @staticmethod
    def name():
        return "hpelmnn"

    def train(self, X, Y):
        class_count = Y.shape[1]
        feature_count = X.shape[1]
        self.__hpelm = ELM(feature_count, class_count, 'wc')
        self.__hpelm.add_neurons(feature_count, "sigm")
        self.__hpelm.train(X, Y)

    def predict(self, X):
        Y_predicted = self.__hpelm.predict(X)
        return Y_predicted