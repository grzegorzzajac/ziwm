from ziwm.data.mock.base import Dataset
import elm
import sys
from os import path
import os

sys.path.append(path.abspath('../../..'))


class TickTackToeDataset(Dataset):
    '''
    First column:
        0: X wins
        1: O wins

    Next 9 columns:
        1|2|3
        -+-+-
        4|5|6
        -+-+-
        7|8|9

        where:
        0: X
        1: O
        2: blank

    example:
        X|X|O      0|0|1
        -+-+-      -+-+-
         |X|O  ->  2|0|1
        -+-+-      -+-+-
        O| |X      1|2|0

    representation in dataset:
        0,0,0,1,2,0,1,1,2,0
    '''

    def name(self):
        return "Tick-Tack-Toe dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data = elm.read("../../ziwm/data/tic_tac_toe/tic_tac_toe.data")
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

