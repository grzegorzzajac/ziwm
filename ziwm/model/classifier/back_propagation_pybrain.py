#!/usr/bin/python2.7

import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer

from ziwm.model.classifier.base import Classifier
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer


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
        # this part converts an activation vector to a class number
        # i'm saving this for a future purpose
        #out = out.argmax(axis=1)  # the highest output activation gives the class
        #if not self.__class_zero_indexing:  # indexing from 1 - add one to result
        #    out += 1
        return out

    def train(self, x, y, class_number=-1):
        self.__class_num = max(np.unique(y).size, class_number)
        if max(y) == self.__class_num:
            self.__class_zero_indexing = False
            y = np.array([i - 1 for i in y])

        DS = ClassificationDataSet(x.shape[1], nb_classes=self.__class_num)
        DS.setField('input', x)
        DS.setField('target', y.reshape(y.size, 1))
        DS._convertToOneOfMany()

        hidden_num = (DS.indim + DS.outdim) / 2

        self.__pybrain_bpnn = buildNetwork(DS.indim, hidden_num, DS.outdim, bias=True, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)

        trainer = BackpropTrainer(self.__pybrain_bpnn, dataset=DS, learningrate=0.07, lrdecay=1.0, momentum=0.6)

        trainer.trainUntilConvergence(DS, maxEpochs=30)
