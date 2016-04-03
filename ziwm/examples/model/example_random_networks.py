#!/usr/bin/python2.7

from ziwm.data.iris.iris import IrisDataset
from ziwm.data.utils import split_dataset
from ziwm.model.ensemble.randomnetworks.random_networks import RandomNetworks
from ziwm.model.utils import round_result
from ziwm.model.base_classifier.extreme_learning_machine.extreme_learning_machine import ExtremeLearningMachine
from ziwm.model.voting_system.arithmetic_mean.arithmetic_mean import ArithmeticMean
from ziwm.model.voting_system.majority_voting.majority_voting import MajorityVoting


elm = []
for i in range(3):
    elm.append(ExtremeLearningMachine())
am = ArithmeticMean()
mv = MajorityVoting()
model = RandomNetworks(elm, mv)
dataset = IrisDataset()
X, Y = dataset.load()
X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

model.train(X_train, Y_train)
prediction = model.predict(X_test)

print 'expected:\n', Y_test
print 'predicted:\n', prediction

#score = model_score(model, X_test, Y_test, problem_type='classification')
#print score