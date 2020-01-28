import os
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from math import log
from enum import Enum

from src.conf.settings import ROOT_DIR, get_logger
from src.io.download import Download
from src.io.extract import Extract
from src.conf.bijective_dict import BijectiveDict

# Logger Setup
logger = get_logger()


class Data:

    def __init__(self, path=None, url=None):
        """

        Parameters
        ----------
        url :
        path :
        name :
        data_dir :
        """
        # Path Defs
        self.url = None
        self.path = None

        if path is None:
            print(
                "Creating an Empty Data Set with no path or root information. "
                "Please specify either url, name or path to proceed.")

        # Data Defs
        self.data = None
        self.train = None
        self.test = None

        if not os.path.isdir(path):
            print("No data directory provided defaulting to " +
                  os.path.join(ROOT_DIR, 'data'))
            self.path = os.path.join(ROOT_DIR, path)
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        else:
            if os.path.isabs(path):
                self.path = path
            else:
                self.path = os.path.join(ROOT_DIR, path)
        if url:
            self.url = url
            self.downloader = Download()
        elif path:
            self.extractor = Extract()

        try:
            self.load()
        except FileNotFoundError as e:
            logger.debug("This is an Exception" + str(e))

        if self.train is None:
            # TODO:Generate Splits
            self.train = self.data
            self.train, self.test = self._train_test_split()

        self.root_dir = os.path.join(path, "model")

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self.mapping = self.prepare_data(self.train)

        for (i, (col_name, col)) in enumerate(self.test.iteritems()):
            self.test[col_name] = col.replace(self.mapping[i])

    def load(self):
        """

        """
        data_dir = self.path
        os.chdir(data_dir)
        for file in os.listdir(data_dir):
            try:
                chunk = pd.read_csv(file, header=None)
                self.data = pd.DataFrame(chunk) if self.data is None else self.data.append(chunk)
            except Exception as e:
                logger.debug("This is an Exception" + str(e))
        os.chdir(ROOT_DIR)

    def set_path(self, path):
        """

        Parameters
        ----------
        path :
        """
        self.path = path

    def set_url(self, url):
        """

        Parameters
        ----------
        url :
        """
        self.url = url

    def _train_test_split(self, ratio=0.8):
        """

        Parameters
        ----------
        ratio :

        Returns
        -------

        """
        n_data = self.data.shape[0]
        mask = np.random.rand(n_data) < ratio
        return self.data[mask], self.data[~mask]

    def vertices(self):
        if self.data is not None:
            vertices = self.data.columns
            return vertices

    def prepare_data(self, data):
        mapping = []
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        transformation_array = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            transformation_array[i] = np.min(data[:,i])
            new_col = data[:,i] - np.min(data[:,i])
            features = np.unique(new_col)
            np.sort(features)
            gap = 0
            for j in range(features.shape[0]-1):
                gap += features[j+1] - features[j]
                if gap > j:
                    new_col[new_col == features[j+1]] -= (gap-j-1)

            mapping.append(BijectiveDict(zip(np.unique(data[:,i]),np.unique(new_col))))
            data[:,i] = new_col
        #for i in range(data.shape[1]):
        #    print(np.unique(data[:,i]).shape[0] ==  np.max(data[:,i]) + 1)

        return mapping

    def radon_number(self, d=None, r=None, h=None, n=None):
        """

            Parameters
            ----------
            :param d: Amount of data points ,i.e. |D|
            :param r: Radon Number -- usually number of model parameters + 2 or number of features +2
            :param h: Number of aggregation steps, we need r models for each aggregation, i.e. we need r**h models for h steps
            :param n: Number of data points per model

            Notes
            -----

        """
        default = False
        n_const = np.sum([1 if item is not None else 0 for item in [d, r, h, n]])
        if n_const == 3:
            default = False
        elif n_const == 4:
            return d, r, h, n
        else:
            default = True

        if not default:
            if h is None:
                h = log(d, r) - log(n, r)
                if h < 1:
                    logger.log(logger.ERROR, "Data is not sufficient for one split. Without additional data"
                                               " or dimensionality reduction, no radon point can be determined.")
                if np.floor(h) == 1:
                    logger.log(logger.WARNING, "Only one splt can be generated i.e. "
                                                 "aggregated for the given amount of data per model.")
                return 'h', h
            elif r is None:
                r = (float(d) / float(n)) ** (1 / float(h))
            elif n is None:
                n = float(d) / float(r) ** h
            else:
                raise ValueError
        else:
            try:
                if d is None:
                    d = self.train.shape[0]
                if r is None:
                    r = self.train.shape[1] + 2
                if n is None:
                    n_candidates = np.arange(1, 1010, 1)
                    h_candidates = np.array([log(d, r) - log(i, r) for i in n_candidates])
                    h = np.floor(np.max(h_candidates))
                    n = n_candidates[np.argmin(h_candidates[h_candidates > h])]
                if h is None:
                    h = log(d, r) - log(n, r)

            except TypeError:
                # TODO ERROR HANDLING
                print("This does not work")

        return d, r, h, n


class GraphType(Enum):
    ChowLiu = 1
    JunctionTree = 2
    Chain = 3
    FullyConnected = 4

