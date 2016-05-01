from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm.classes import LinearSVC

from ziwm.model.classifier.base import Classifier

class SVM(Classifier):

    def __init__(self):
        self.__clf = OneVsRestClassifier(LinearSVC())
    
    @staticmethod
    def name():
        return "svm"
    
    def train(self, X, Y):
        self.__clf.fit(X, Y)
        
    def predict(self, X):
        return self.__clf.predict(X)