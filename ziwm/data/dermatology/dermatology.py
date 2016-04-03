#!/usr/bin/python2.7

import elm
from os import path
from ziwm.data.mock.base import Dataset


class Dermatology(Dataset):
    '''
    Class code:   Class:                  Number of instances:
       1             psoriasis			            112
       2             seboreic dermatitis             61
       3             lichen planus                   72
       4             pityriasis rosea                49
       5             cronic dermatitis               52
       6             pityriasis rubra pilaris        20

    Clinical Attributes: (take values 0, 1, 2, 3, unless otherwise indicated)
          1: erythema
          2: scaling
          3: definite borders
          4: itching
          5: koebner phenomenon
          6: polygonal papules
          7: follicular papules
          8: oral mucosal involvement
          9: knee and elbow involvement
         10: scalp involvement
         11: family history, (0 or 1)
         34: Age (linear)

     Histopathological Attributes: (take values 0, 1, 2, 3)
         12: melanin incontinence
         13: eosinophils in the infiltrate
         14: PNL infiltrate
         15: fibrosis of the papillary dermis
         16: exocytosis
         17: acanthosis
         18: hyperkeratosis
         19: parakeratosis
         20: clubbing of the rete ridges
         21: elongation of the rete ridges
         22: thinning of the suprapapillary epidermis
         23: spongiform pustule
         24: munro microabcess
         25: focal hypergranulosis
         26: disappearance of the granular layer
         27: vacuolisation and damage of basal layer
         28: spongiosis
         29: saw-tooth appearance of retes
         30: follicular horn plug
         31: perifollicular parakeratosis
         32: inflammatory monoluclear inflitrate
         33: band-like infiltrate
    '''

    def name(self):
        return "Dermatology dataset"

    def problem_type(self):
        return "classification"

    def load(self):
        data_path = path.join(path.dirname(path.abspath(__file__)), "dermatology.data")
        data = elm.read(data_path)
        x = data[:, 1:]
        y = data[:, 0]
        return x, y

