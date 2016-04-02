from ziwm.data.house_votes_84.house_votes_84 import HouseVotes84
from ziwm.data.utils import split_dataset
from ziwm.model.ensemble.bagging.bagging import Bagging
from ziwm.model.utils import round_result
from ziwm.model.base_classifier.extreme_learning_machine.extreme_learning_machine import ExtremeLearningMachine
from ziwm.model.voting_system.arithmetic_mean.arithmetic_mean import ArithmeticMean
from ziwm.model.voting_system.majority_voting.majority_voting import MajorityVoting
import numpy as np
from ziwm.benchmarks.ziwm_tool import model_score

elm = []
for i in range(39):
    elm.append(ExtremeLearningMachine())
am = ArithmeticMean()
mv = MajorityVoting()
model = Bagging(elm, mv)
hv84 = HouseVotes84()
X, Y = hv84.load()
X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

model.train(X_train, Y_train)
prediction = model.predict(X_test)

print 'input:\n', X_test
print 'expected output:\n', Y_test
print 'predicted output:\n', prediction

score = model_score(model, X_test, Y_test, problem_type='classification')
print 'score: ', score
