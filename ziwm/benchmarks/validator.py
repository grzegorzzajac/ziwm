#!/usr/bin/python2.7

import numpy as np
from ziwm.model.base_classifier.mock_classifier.mock_classifier import MockClassifier
from ziwm.data.utils import split_dataset

def __average_square_error(Y1, Y2):
    length = Y1.shape[0]
    assert Y1.shape == Y2.shape
    
    error = np.sum((Y1 - Y2) ** 2.0)
    error = error / (1.0 * length)
    return error

def __regression_error(Y1, Y2):
    return __average_square_error(Y1, Y2)

def __classification_error(Y_predicted, Y):
    assert Y_predicted.shape == Y.shape
    length = Y_predicted.shape[0]

    Y_class = np.rint(Y).astype(int)
    Y_pred_class = np.rint(Y_predicted).astype(int)

    error = ((Y_pred_class != Y_class) * 1.).sum()
    error = error / float(length)
    return error

def model_score(model, X_test, Y_test, problem_type):
    '''
    Calculates performance score of a model on a given examples set
    '''
    Y_predicted = model.predict(X_test)
    assert Y_predicted.shape == Y_test.shape

    if problem_type == 'regression':
        score = 1.0 - __regression_error(Y_predicted, Y_test)
    elif problem_type == 'classification':
        score = 1.0 - __classification_error(Y_predicted, Y_test)
    else:
        raise Exception("problem_type not recognized", problem_type)
    
    return score

if __name__ == '__main__':
    model = MockClassifier()
    dataset = MockDataset()
    X, Y = dataset.load()
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
    model.train(X_train, Y_train)
    score = model_score(model, X_test, Y_test, problem_type='classification')
    print(score)
