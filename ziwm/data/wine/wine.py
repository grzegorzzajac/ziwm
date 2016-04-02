from ziwm.data.mock.base import Dataset
import elm
import sys
from os import path
import os

sys.path.append(path.abspath('../../..'))


class Wine(Dataset):
    '''

    '''

    def name(self):
        return "Wine dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data = elm.read("../../ziwm/data/wine/wine.data")
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

