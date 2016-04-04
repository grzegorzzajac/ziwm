#!/usr/bin/python2.7

from ziwm.benchmarks.validator import model_score
from ziwm.data.iris.iris import IrisDataset
from ziwm.data.breast_cancer.breast_cancer import BreastCancer
from ziwm.data.tic_tac_toe.tic_tac_toe import TickTackToeDataset
from ziwm.data.utils import split_dataset
from ziwm.model.base_classifier.back_propagation_neurolab.back_propagation_neurolab import BackPropagationNeuroLab
from ziwm.model.base_classifier.back_propagation_pybrain.back_propagation_pybrain import BackPropagationPyBrain


model_neurolab = BackPropagationNeuroLab()
model_pybrain = BackPropagationPyBrain()
dataset = IrisDataset()

X, Y = dataset.load()
X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

model_neurolab.train(X_train, Y_train)
model_pybrain.train(X_train, Y_train)

print Y_test
print model_neurolab.predict(X_test)
print model_pybrain.predict(X_test)

#score = model_score(model, X_test, Y_test, problem_type='classification')
#print score
