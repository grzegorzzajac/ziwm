from ziwm.data.iris.iris import IrisDataset
from ziwm.model.elmk.elmk import ExtremeLearningMachine
from ziwm.data.utils import split_dataset
from ziwm.validator import model_score
import sys
from os import path
import numpy as np
from sklearn.cross_validation import train_test_split

sys.path.append(path.abspath('..'))

model1 = ExtremeLearningMachine()
model2 = ExtremeLearningMachine()
dataset = IrisDataset()

X, Y = dataset.load()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80, random_state=2)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y, test_size=0.80, random_state=3)

print "Dane uczace: "
print X_train, Y_train
print '<<<<<<<<>>>>>>>>'
print X_train2, Y_train2

model1.train(X_train, Y_train)
model2.train(X_train2, Y_train2)

print 'Oczekiwane wyniki:'
print Y_test
print 'Siec 1 wyniki: '
r1 = model1.predict(X_test)
print r1
print 'Siec 2 wyniki: '
r2 = model2.predict(X_test)
print r2

if (r1 == r2).all():
    print 'wyniki takie same'
else:
    print 'wyniki inne'

if (X_train == X_train2).all():
    print 'dane uczace takie same'
else:
    print 'dane uczace inne'

if model1 == model2:
    print 'modele takie same'
else:
    print 'modele inne'