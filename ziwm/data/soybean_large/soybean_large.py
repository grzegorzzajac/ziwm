from ziwm.data.mock.base import Dataset
import elm
import sys
from os import path
import os

sys.path.append(path.abspath('../../..'))


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
        return "Soybean Small dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data = elm.read("../../ziwm/data/soybean_large/soybean_large.data")
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

