#!/usr/bin/python2.7

import elm
from os import path
from ziwm.data.base import Dataset


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
        data_path = path.join(path.dirname(path.abspath(__file__)), "house_votes_84.data")
        data = elm.read(data_path)
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

