import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm.classes import SVC

from ziwm.model.classifier.base import Classifier

class SVM(Classifier):

    def __init__(self):
        self.__class_zero_indexing = True
        self.__class_num = 0
        self.__clf = OneVsRestClassifier(SVC(probability=True))
    
    @staticmethod
    def name():
        return "svm"
    
    def train(self, X, Y, class_number=-1):
        self.__class_num = max(np.unique(Y).size, class_number)
        self.__clf.fit(X, Y)

    def predict(self, X):
        out = self.__clf.predict_proba(X)
        assert len(out[0]) == self.__class_num
        return out

    def predict2(self, X):
        out = self.__clf.predict(X)
        # assert len(out[0]) == self.__class_num
        return out