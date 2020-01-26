#!/usr/bin/env python

"""
TODO: Description Here
"""
import os

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

from .dataset import Data

from src.conf.settings import ROOT_DIR, DEBUG, get_logger

# Logger Setup
logger = get_logger()


class Susy(Data):

    def __init__(self, url=None, path=None, name=None, data_dir=None):

        self._disc_opts = {Discretization.Quantile: self._quantile,
                           Discretization.KMeans: self._kmeans,
                           Discretization.Distance: self._distance

                           }

        super(Susy, self).__init__(url, path, name, data_dir)

    def discretize(self, opt):
        for (col_name, col) in self.train.iteritems():
            if self._is_numeric(col):
                #self.train[col_name] = self._disc_opts[opt](col)
                self.train[col_name] = self._quantile(col)

    def load(self):
        """

        """
        data_dir = os.path.join(self.path, self.name)
        os.chdir(data_dir)
        for file in os.listdir(data_dir):
            try:
                dataframe_generator = pd.read_csv(file, header=None, chunksize=10000)
                for chunk in dataframe_generator:
                    self.data = pd.DataFrame(chunk) if self.data is None else self.data.append(chunk)
                    if DEBUG:
                        break
            except Exception as e:
                logger.debug("This is an Exception" + str(e))
        os.chdir(ROOT_DIR)

    def _is_numeric(self, col):
        return True

    def _quantile(self, col):
        data = col.to_numpy()
        arr = np.linspace(0,1,10)
        quantiles = np.quantile(col, arr)
        dist = np.zeros(shape=(data.shape[0]))
        for i, point in enumerate(data):
            dist[i] = quantiles[np.argmin(np.abs(quantiles - point))]
        return pd.Series(dist)


    def _kmeans(self, col):
        pass

    def _distance(self, col):
        pass


from enum import Enum


class Discretization(Enum):
    Quantile = 1
    KMeans = 2
    Distance = 3


