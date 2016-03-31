from ziwm.data.iris.iris import IrisDataset
from ziwm.model.elmk.elmk import ExtremeLearningMachine
from ziwm.data.utils import split_dataset
from ziwm.model.utils import round_result
from ziwm.validator import model_score
import sys
from os import path
import numpy as np


sys.path.append(path.abspath('..'))

model = ExtremeLearningMachine()
dataset = IrisDataset()

X, Y = dataset.load()
X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

model.train(X_train, Y_train)

print Y_test
print model.predict(X_test)

score = model_score(model, X_test, Y_test, problem_type='classification')
print score
