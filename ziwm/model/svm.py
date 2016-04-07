from ziwm.model.base_classifier.classifier import Classifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm.classes import LinearSVC

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