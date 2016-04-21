#!/usr/bin/python2.7

import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from ziwm.model.base_classifier.classifier import Classifier


class BackPropagationPyBrain(Classifier):
    '''
    Description...
    '''
    def __init__(self):
        self.__class_zero_indexing = True

    @staticmethod
    def name():
        return "Back Propagation PyBrain"

    def predict(self, x_test):
        DS = ClassificationDataSet(x_test.shape[1], nb_classes=self.__class_num)
        DS.setField('input', x_test)
        DS.setField('target', np.zeros((x_test.shape[0], 1)))
        DS._convertToOneOfMany()
        out = self.__pybrain_bpnn.activateOnDataset(DS)
        out = out.argmax(axis=1)  # the highest output activation gives the class
        if not self.__class_zero_indexing:  # indexing from 1 - add one to result
            out += 1
        return out

    def train(self, x, y):
        self.__class_num = np.unique(y).size
        if max(y) == self.__class_num:
            self.__class_zero_indexing = False
            y = np.array([i - 1 for i in y])

        DS = ClassificationDataSet(x.shape[1], nb_classes=self.__class_num)
        DS.setField('input', x)
        DS.setField('target', y.reshape(y.size, 1))
        DS._convertToOneOfMany()

        hidden_num = (DS.indim + DS.outdim) / 2

        self.__pybrain_bpnn = buildNetwork(DS.indim, hidden_num, DS.outdim, outclass=SoftmaxLayer)

        trainer = BackpropTrainer(self.__pybrain_bpnn, dataset=DS)

        trainer.trainEpochs(5)
