from ziwm.data.mock.base import Dataset
import elm
import sys
from os import path
import os

sys.path.append(path.abspath('../../..'))


class SoybeanSmall(Dataset):
    '''
    Description...
    '''

    def name(self):
        return "Soybean Small dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data = elm.read("../../ziwm/data/soybean_small/soybean_small.data")
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

