import os
import json
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import scipy.stats
import pxpy as px

from math import log
from enum import Enum

from src.conf.settings import CONFIG
from src.io.download import Download
from src.io.extract import Extract
from src.conf.bijective_dict import BijectiveDict

# Logger Setup
logger = CONFIG.get_logger()
ROOT_DIR = CONFIG.ROOT_DIR


class Discretization(Enum):
    Quantile = 1
    KMeans = 2
    Distance = 3


class GraphType(Enum):
    ChowLiu = 1
    JunctionTree = 2
    Chain = 3
    FullyConnected = 4


class Data:

    def __init__(self, path=None, url=None, mask=None, seed=None, cval=10):
        """

        Parameters
        ----------
        url :
        path :
        """

        self.random_state = np.random.RandomState(seed=seed)
        # Path Defs
        self.url = None
        self.path = None
        self.seed = seed

        if path is None:
            print(
                "Creating an Empty Data Set with no path or root information. "
                "Please specify either url, name or path to proceed.")

        # Data Defs

        self.features_dropped = False
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
            self.downloader.start()
            self.extractor = Extract()
            self.extractor.extract_all()
        elif path:
            self.extractor = Extract()

        self.data = None

        self.train = None
        self.test = None
        self.test_labels = None
        self.holdout = None
        self.split = None
        self.holdout_size = 10000

        self.mask = mask
        self.masks = []
        self.bins = {}
        self.label_column = 0
        self.cv = cval
        self.disc_quantiles = 10
        self.root_dir = os.path.join(path, "model")

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

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
                logger.debug("Unable to load file : " + str(file) + "\nException :" + str(e))
        os.chdir(ROOT_DIR)

    def _train_test_split(self, ratio=0.8, holdout_size=10000):
        """

        Parameters
        ----------
        ratio :

        Returns
        -------

        """
        n_data = self.data.shape[0]
        mask = self.random_state.rand(n_data) < ratio
        return mask, self.data[mask][holdout_size:], self.data[~mask], self.data[mask][0:holdout_size]

    def create_cv_split(self):
        instances, variables = self.data.shape
        index = np.arange(instances)
        self.random_state.shuffle(index)
        self.holdout = self.data.iloc[index[:self.holdout_size]]
        target_columns = np.delete(np.arange(self.holdout.shape[1]), self.label_column)
        holdout_disc, _ = px.discretize(np.ascontiguousarray(self.holdout.to_numpy()), num_states=self.disc_quantiles, targets=target_columns)
        for (col_name, col) in self.holdout.iteritems():
            if col_name != self.label_column:
                self.holdout.loc[::,col_name] = holdout_disc[:,col_name]
        self.split = np.array_split(index[self.holdout_size:], self.cv)

    def reset_cv(self):
        self.load_cv_split(0)

    def load_cv_split(self, i):
        n_splits = np.arange(self.cv)
        train = n_splits[np.arange(self.cv) != i]
        self.test = self.data.iloc[self.split[i]]
        self.train = self.data.iloc[np.concatenate(np.array(self.split)[train])]

        new_mask = np.zeros(self.data.shape[0], dtype=np.bool)
        new_mask[np.concatenate(np.array(self.split)[train])] = True
        self.masks.append(new_mask)
        self.mask = new_mask

        target_columns = np.delete(np.arange(self.train.shape[1]), self.label_column)
        train_disc, disc_map = px.discretize(np.ascontiguousarray(self.train.values), num_states=self.disc_quantiles, targets=target_columns)
        test_disc, _ = px.discretize(np.ascontiguousarray(self.test.values), discretization=disc_map, targets=target_columns)

        for (col_name, col) in self.train.iteritems():
            if col_name != self.label_column:
                self.train.loc[::, col_name] = train_disc[:,col_name]
                self.test.loc[::, col_name] = test_disc[:,col_name]

        self.test_labels = np.copy(self.test[self.label_column].to_numpy())

    def discretize(self):
        quants = np.linspace(0, 1, 9)
        for (col_name, col) in self.train.iteritems():
            if col_name != self.label_column and np.unique(col).shape[0] > self.disc_quantiles:
                _, bins = pd.qcut(col, quants, labels=False, retbins=True, duplicates='drop')
                self.train.loc[:, col_name] = np.digitize(self.train[col_name], bins)
                self.test.loc[:, col_name] = np.digitize(self.test[col_name], bins)
                self.holdout.loc[:, col_name] = np.digitize(self.holdout[col_name], bins)
                self.bins[col_name] = bins

    def vertices(self):
        if self.data is not None:
            vertices = self.data.columns
            return vertices

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

    def prepare_data(self, data):
        mapping = []
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        transformation_array = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            transformation_array[i] = np.min(data[:, i])
            new_col = data[:, i] - np.min(data[:, i])
            features = np.unique(new_col)
            np.sort(features)
            gap = 0
            for j in range(features.shape[0] - 1):
                gap += features[j + 1] - features[j]
                if gap > j:
                    new_col[new_col == features[j + 1]] -= (gap - j - 1)

            mapping.append(BijectiveDict(zip(np.unique(data[:, i]), np.unique(new_col))))
            data[:, i] = new_col
        # for i in range(data.shape[1]):
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


class Synthetic(Data):

    def __init__(self, states, edgelist=None, seed=None):
        super(Synthetic, self).__init__()
        n_vars = 15
        n_samples = 1000
        n_states = 10
        self.random_state = np.random.RandomState(seed=seed)
        # Generate random cov
        cov = self.random_state.randn(n_vars, n_vars)
        cov = np.dot(cov, cov.T) / n_vars

        # Generate data from normal
        self.data = pd.DataFrame(scipy.stats.multivariate_normal(mean=np.zeros(n_vars), cov=np.dot(cov, cov.T) / n_vars).rvs(n_samples))

        data_disc, disc_ttt = px.discretize(data= self.data, num_states=n_states)

        # Add sample to ensure same state space for each variable
        data_disc = np.concatenate([data_disc, np.full(shape=(1, n_vars), fill_value=n_states - 1, dtype=np.uint16)])

        # Generate model
        self.global_model = px.train(data_disc, graph=px.GraphType.auto_tree, mode=px.ModelType.mrf, iters=0)
        self.global_weights = np.copy(self.global_model.weights)
        # TODO: Remove the statistics for full point.

        edgelist = self.global_model.graph.edgelist
        stats = self.global_model.statistics

    def set_weights(self, weights):
        np.copyto(self.global_model.weights, weights)

    def ll(self):
        mu, A = self.global_model.infer()
        return - (np.inner(self.global_model.statistics, self.global_model.weights) - A)

    def load_cv_split(self, i, ratio):
        n_splits = np.arange(self.cv)
        train = n_splits[np.arange(self.cv) != i]
        self.test = self.data.iloc[self.split[i]]
        self.train = self.data.iloc[np.concatenate(np.array(self.split)[train])]
        new_mask = np.zeros(self.data.shape[0], dtype=np.bool)
        new_mask[np.concatenate(np.array(self.split)[train])] = True
        self.masks.append(new_mask)

        self.mask = new_mask

        train_disc, disc_map = px.discretize(self.train.values)
        test_disc, _ = px.discretize(self.test.values, discretization=disc_map)

        self.train[:] = train_disc
        self.test[:] = test_disc

        self.test_labels = np.copy(self.test[self.label_column].to_numpy())


class Dota2(Data):

    def __init__(self, url=None, path=None, mask=None, seed=None,
                 train_test_ratio=0.9,
                 discretization_quantiles=9,
                 label_col=0,
                 cval=10):
        super(Dota2, self).__init__(path, url, mask, seed, cval=cval)

        self.heroes = {}
        self.hero_list = None

        self.label_column = label_col
        self.holdout_size = 10000

        self.ratio = train_test_ratio
        self.holdout_size = 10000
        self.validation = None
        try:
            self.load()
        except FileNotFoundError as fe:
            raise RuntimeError("Beep Boop We didn't find the file you were looking for.")
        self.load_json()
        self.mapping = self.prepare_data(self.data)

        for (i, (col_name, col)) in enumerate(self.data.iteritems()):
            self.validation.loc[:,col_name] = col.replace(self.mapping[i])

    def load(self):
        """
        Dota2 has a separate Test/Validation set.
        """
        data_dir = self.path
        os.chdir(data_dir)
        train = "dota2Train.csv"
        test  = "dota2Test.csv"
        try:
            self.data = pd.read_csv(train, header=None)
            self.validation = pd.read_csv(test, header=None)
        except Exception as e:
            logger.debug("Unable to load file : " + "\nException :" + str(e))
        os.chdir(ROOT_DIR)

    def load_json(self):
        with open(os.path.join(ROOT_DIR, 'data', 'DOTA2', 'heroes.json')) as file:
            data = json.load(file)

            heroes = list(data.values())
            self.heroes = {i['id']: i['name'] for i in heroes[0]}
            ordered_heroes = list(range(len(self.heroes) + 1))
            for i, (id, name) in enumerate(self.heroes.items()):
                ordered_heroes[id - 1] = name
            self.hero_list = ordered_heroes

    def drop(self, cols):
        self.data = self.data.drop(columns=cols)
        self.train = self.train.drop(columns=cols)
        self.test = self.test.drop(columns=cols)
        self.holdout = self.holdout.drop(columns=cols)
        # self.features_dropped = True

    def data_header(self):
        header = ['Result', 'ClusterID', 'GameMode', 'GameType'] + self.hero_list
        self.data.reset_index()
        self.data.columns = header
        self.validation.reset_index()
        self.validation.columns = header

    def sample_match(self):
        matches = self.data.shape[0]
        match_id = np.random.randint(0, high=matches - 1)

        match = self.data.iloc[match_id]

        result = match['Result']
        clusterid = match['ClusterID']
        gamemode = match['GameMode']
        gametype = match['GameType']

        picks = match.drop(labels=['Result', 'ClusterID', 'GameMode', 'GameType'])
        radiant = picks.where(lambda x: x == -1).dropna()
        dire = picks.where(lambda x: x == 1).dropna()

        if result == 1:
            winner = "Dire"
            print(winner + " won the match with \n" + str(list(dire.index)) + " against the radiant with \n" + str(
                list(radiant.index)))
        else:
            winner = "Radiant"
            print(winner + " won the match with \n" + str(list(radiant.index)) + " against the dire with \n" + str(
                list(dire.index)))

    def to_csv(self, path):
        self.train.to_csv(path)

    def __name__(self):
        return self.__class__.__name__


class Susy(Data):

    def __init__(self, url=None, path=None, mask=None, seed=None, cval=10):

        super(Susy, self).__init__(path, url, mask, seed, cval=cval)

        if self.random_state is None:
            self.random_state = np.random.RandomState(seed=seed)

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

        self.quantiles = BijectiveDict()
        self.label_column = 0
        self.holdout_size = 10000

        self.ratio = 0.9

        try:
            self.load()
        except FileNotFoundError as e:
            logger.debug("Couldn't load a file in the folder: " + str(e))

    def load(self):
        """

        """
        data_dir = self.path
        os.chdir(data_dir)
        for file in os.listdir(data_dir):
            try:
                dataframe_generator = pd.read_csv(file, header=None, chunksize=1000000)
                for chunk in dataframe_generator:
                    self.data = pd.DataFrame(chunk) if self.data is None else self.data.append(chunk)
            except Exception as e:
                logger.debug("This is an Exception" + str(e))
        os.chdir(ROOT_DIR)

    def _quantile(self, col):
        data = col.to_numpy()
        arr = np.linspace(0, 1, 10)
        quantiles = np.quantile(col, arr)
        dist = np.zeros(shape=(data.shape[0]))
        for i, point in enumerate(data):
            dist[i] = np.argmin(np.abs(quantiles - point))
        return quantiles, pd.Series(dist)

    def _kmeans(self, col):
        pass

    def _distance(self, col):
        pass


class CoverType(Data):

    def __init__(self, url=None, path=None, mask=None, seed=None,
                 train_test_ratio=0.9,
                 discretization_quantiles=9,
                 label_col=54,
                 cval=10):
        super(CoverType, self).__init__(path=path, url=url, mask=mask, seed=seed, cval=cval)

        if self.random_state is None:
            self.random_state = np.random.RandomState(seed=seed)

        self.ratio = train_test_ratio
        self.holdout_size = 10000
        self.quantiles = discretization_quantiles

        self.label_column = label_col


        try:
            self.load()
        except FileNotFoundError as e:
            logger.debug("Couldn't load a file in the folder: " + str(e))

        self.prep_folder(path)
        self.data[self.label_column] -= 1

    def prep_folder(self, path):
        if not os.path.isdir(path):
            logger.info("No data directory provided defaulting to " +
                        os.path.join(ROOT_DIR, 'data'))
            self.path = os.path.join(ROOT_DIR, path)
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
        else:
            if os.path.isabs(path):
                self.path = path
            else:
                self.path = os.path.join(ROOT_DIR, path)


class Fact(Data):

    def __init__(self, url=None, path=None, mask=None, seed=None):
        super(Fact, self).__init__(url, path, mask, seed)
