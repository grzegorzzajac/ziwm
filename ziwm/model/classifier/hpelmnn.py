from ziwm.model.classifier.base import Classifier

from hpelm import ELM
import numpy as np
from sklearn.preprocessing.data import OneHotEncoder
import sys
import os


class HPELMNN(Classifier):
    
    def __init__(self):
        self.__hpelm = None
    
    @staticmethod
    def name():
        return "hpelmnn"

    def train(self, X, Y, class_number=-1):
        class_count = max(np.unique(Y).size, class_number)
        feature_count = X.shape[1]
        self.__hpelm = ELM(feature_count, class_count, 'wc')
        self.__hpelm.add_neurons(feature_count, "sigm")

        Y_arr = Y.reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(Y_arr)
        Y_OHE = enc.transform(Y_arr).toarray()

        out_fd = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self.__hpelm.train(X, Y_OHE)
        sys.stdout = out_fd

    def predict(self, X):
        Y_predicted = self.__hpelm.predict(X)
        return Y_predicted