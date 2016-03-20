from ziwm.data.dataset.base import Dataset
import numpy as np


class MockDataset(Dataset):
    '''
    A simple, meaningless, mock dataset.
    '''
    
    def name(self):
        return "mock_dataset"
    
    def problem_type(self):
        return "classification"

    def load(self):
        X = np.array([[0., 0., 0.],
                      [0., 0., 1.],
                      [0., 1., 0.],
                      [0., 1., 1.],
                      [1., 0., 0.],
                      [1., 0., 1.],
                      [1., 1., 0.],
                      [1., 1., 1.]])
        Y = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
        return X, Y