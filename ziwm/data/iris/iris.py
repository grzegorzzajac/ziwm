from ziwm.data.mock.base import Dataset
import elm
import sys
from os import path
import os

sys.path.append(path.abspath('../../..'))


class IrisDataset(Dataset):
    '''
    Iris dataset: https://raw.githubusercontent.com/acba/extreme_learning_machine/develop/tests/data/iris.data
    '''

    def name(self):
        return "Iris dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data = elm.read("../../ziwm/data/iris/iris.data")
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

