#!/usr/bin/python2.7

import elm
from os import path
from ziwm.data.base import Dataset


class IrisDataset(Dataset):
    '''
    Iris dataset: https://raw.githubusercontent.com/acba/extreme_learning_machine/develop/tests/data/iris.data
    '''

    def name(self):
        return "Iris dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data_path = path.join(path.dirname(path.abspath(__file__)), "iris.data")
        data = elm.read(data_path)
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

