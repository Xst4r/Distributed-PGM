#!/usr/bin/env python

"""
TODO: Description Here
"""
import os

import pandas as pd
import numpy as np

from src.conf.settings import ROOT_DIR, get_logger
from src.model.aggregation import radon_machine
# Logger Setup
logger = get_logger()



def main():
    r = np.random.randint(100, 300)
    h = 2
    print("Allocating " + str((r**h * (r-2) * 8)/(1e6)) + " MB Memory")
    weights = np.random.normal(loc= 20, scale=50, size=(r-2) * r**h).reshape(r-2, r**h)

    radon_point = radon_machine(weights, r, h)
    print(radon_point.shape[0] == r-2)
    return r, h, weights, radon_point

if __name__ == '__main__':
    r, h, weights, radon_point = main()

