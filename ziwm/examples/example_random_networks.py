from ziwm.data.iris.iris import IrisDataset
from ziwm.data.utils import split_dataset
from ziwm.model.ensemble.randomnetworks.random_networks import RandomNetworks
from ziwm.model.utils import round_result

model = RandomNetworks(2)
dataset = IrisDataset()

X, Y = dataset.load()
X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

model.train(X_train, Y_train)

print 'expected:\n', Y_test
print 'predicted:\n', round_result(model.predict(X_test))

#score = model_score(model, X_test, Y_test, problem_type='classification')
#print score