from ziwm.model.base import Model

import numpy as np

class MockClassifier(Model):
    '''
    A simple, mock classifier. It does not learn.
    Always returns '1' if all features are equal to '1'
    and '0' otherwise
    '''
    
    def name(self):
        return "MockClassifier"
    
    def predict(self, X):
        '''
        Predicts '1' if all features are equal to '1',
        '0' otherwise
        '''
        row_mins = X.min(axis=1)
        row_maxs = X.max(axis=1)
        Y = ((row_mins == 1) & (row_maxs == 1))
        Y = Y * 1.0
        return Y
    
    def train(self, X, Y):
        '''
        Do nothing - it is just a mock
        '''
        pass
    
if __name__ == "__main__":

    X = np.arange(50).reshape(10,5) / 50.0
    X = np.vstack((X, np.ones(5)))
    X = np.vstack((np.ones(5), X))
    print("X:\n{}".format(X))

    model = MockClassifier()
    model.train(X, None)
    Y_predicted = model.predict(X)
    
    print("predicted Y:\n{}".format(Y_predicted))

    print("Done")
    