#!/usr/bin/python2.7

import numpy as np
from sklearn.cross_validation import train_test_split
from ziwm.data.iris.iris import IrisDataset
from ziwm.model.base_classifier.extreme_learning_machine.extreme_learning_machine import ExtremeLearningMachine


model1 = ExtremeLearningMachine()
model2 = ExtremeLearningMachine()
dataset = IrisDataset()

X, Y = dataset.load()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=2)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.20, random_state=3)

print "Dane uczace: "
print X_train, Y_train
print '<<<<<<<<>>>>>>>>'
print X_train2, Y_train2

model1.train(X_train, Y_train)
model2.train(X_train, Y_train)

print 'Siec 1 wyniki: '
r1 = model1.predict(X_test)
print r1
print 'Siec 2 wyniki: '
r2 = model2.predict(X_test)
print r2

results = [r1, r2]
avg = np.average(results, axis=0)
print 'Usrednione wyniki'
print abs(avg.round(0))

print 'Oczekiwane wyniki:'
print Y_test