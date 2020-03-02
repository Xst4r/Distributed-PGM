#!/usr/bin/env python

"""
TODO: Description Here
"""
import os

import pandas as pd
import numpy as np

from src.conf.settings import ROOT_DIR, get_logger
from src.model.aggregation import RadonMachine
# Logger Setup
logger = get_logger()



def main():
    r = np.random.randint(3, 4)
    h = 10
    mu = np.random.rand(r-2)
    cov = np.random.rand((r-2)**2).reshape(r-2, r-2)
    print("Allocating " + str((r**h * (r-2) * 8)/(1e6)) + " MB Memory")
    weights = np.random.multivariate_normal(mean=mu, cov=cov, size=r**h).T

    radon_point = RadonMachine(weights, r, h)
    radon_point.aggregate(None)
    #print(radon_point.shape[0] == r-2)
    return r, h, weights, radon_point

if __name__ == '__main__':
    r, h, weights, radon_point = main()

