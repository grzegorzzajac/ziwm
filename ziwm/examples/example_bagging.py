from ziwm.data.iris.iris import IrisDataset
from ziwm.data.utils import split_dataset
from ziwm.model.ensemble.bagging.bagging import Bagging
from ziwm.model.utils import round_result
from ziwm.model.base_classifier.extreme_learning_machine.extreme_learning_machine import ExtremeLearningMachine

elm = [ExtremeLearningMachine()] * 3
model = Bagging(elm, 5)
dataset = IrisDataset()

X, Y = dataset.load()
X_train, X_test, Y_train, Y_test = split_dataset(X, Y)


model.train(X_train, Y_train)

print 'expected:\n', Y_test
print 'predicted:\n', round_result(model.predict(X_test))

#score = model_score(model, X_test, Y_test, problem_type='classification')
#print score