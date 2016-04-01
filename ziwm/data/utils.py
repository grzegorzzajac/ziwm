#!/usr/bin/python2

import numpy as np
from sklearn.cross_validation import train_test_split

def split_dataset(X, Y):
    '''
    Split mock_classifier into train and examples set.
    Returned mock_classifier is ordered randomly but always the same.
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1993)
    return X_train, X_test, Y_train, Y_test

    
    
if __name__ == "__main__":
    X = np.arange(50).reshape(10,5)
    Y = np.arange(10).reshape(10,1)
    print("Splitting mock_classifier:")
    print("X:\n{}".format(X))
    print("Y:\n{}".format(Y))

    print("Into train and examples sets:")
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
    print("X train:\n{}".format(X_train))
    print("X examples:\n{}".format(X_test))
    print("Y train:\n{}".format(Y_train))
    print("Y examples:\n{}".format(Y_test))
