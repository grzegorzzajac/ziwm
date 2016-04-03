#!/usr/bin/python2.7

import elm
from os import path
from ziwm.data.mock.base import Dataset


class SoybeanSmall(Dataset):
    '''
    Description...
    '''

    def name(self):
        return "Soybean Small dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data_path = path.join(path.dirname(path.abspath(__file__)), "soybean_small.data")
        data = elm.read(data_path)
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

