from ziwm.data.dataset.base import Dataset
import elm
import sys
from os import path

sys.path.append(path.abspath('../..'))


class IrisDataset(Dataset):
    '''
    Iris dataset: https://raw.githubusercontent.com/acba/pelm/develop/tests/data/iris.data
    '''

    def name(self):
        return "Iris dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data = elm.read("data/iris/iris.data")
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

