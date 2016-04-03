#!/usr/bin/python2.7

import elm
from os import path
from ziwm.data.mock.base import Dataset


class SoybeanLarge(Dataset):
    '''
    0: diaporthe-stem-canker
    1: charcoal-rot
    2: rhizoctonia-root-rot
    3: brown-stem-rot
    4: powdery-mildew
    5: downy-mildew
    6: brown-spot
    7: bacterial-blight
    8: bacterial-pustule
    9: purple-seed-stain
    10: anthracnose
    11: phyllosticta-leaf-spot
    12: alternarialeaf-spot
    13: frog-eye-leaf-spot
    '''

    def name(self):
        return "Soybean Large dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data_path = path.join(path.dirname(path.abspath(__file__)), "soybean_large.data")
        data = elm.read(data_path)
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

