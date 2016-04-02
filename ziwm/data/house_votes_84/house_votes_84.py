from ziwm.data.mock.base import Dataset
import elm
import sys
from os import path
import os

sys.path.append(path.abspath('../../..'))


class HouseVotes84(Dataset):
    '''
    Class:
        0: democrat
        1: republican

    Attributes:
        0: yes
        1: not yes, not no
        2: no
    '''

    def name(self):
        return "House Votes 1984 dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data = elm.read("../../ziwm/data/house_votes_84/house_votes_84.data")
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

