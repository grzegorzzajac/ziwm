#!/usr/bin/python2.7

import elm
from os import path
from ziwm.data.base import Dataset


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
        data_path = path.join(path.dirname(path.abspath(__file__)), "tic_tac_toe.data")
        data = elm.read(data_path)
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

