import sys
from os import path
from ziwm.benchmarks.validator import model_score
from ziwm.data.iris.iris import IrisDataset
from ziwm.data.utils import split_dataset
from ziwm.model.base_classifier.extreme_learning_machine.extreme_learning_machine import ExtremeLearningMachine

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
