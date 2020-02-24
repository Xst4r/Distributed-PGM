import logging
from math import log

import numpy as np

from enum import Enum
from src.model.aggregation import AggregationType


class SplitType(Enum):
    Random = 1
    Bootstrap = 2
    Model = 3


class Sampler:

    def __init__(self, data, n_splits=10, sample_complexity=100, seed=None, k=10):
        """
            Parameters
            ----------
            n_splits : Integer
                Number of Devices to distribute the data onto, i.e., the number of Splits to be generated

            See Also
            --------

            Notes
            -----
            This Class manages the data splits and is usually called when calling the train method in any PGM Object.
            You can create your own Split Object and pass it to the PGM Object. Otherwise, a default Split Object with a random split
            using 10 devices will be created. You may adjust the default split at any time by calling the respective methods.

            Examples
            --------
            from src.conf.modes.splits import Splits

            my_split = Splits(devices = 100)
            my_split.set_mode(Splits.corr)
            my_split.create_split()

        """

        self.random_state = np.random.RandomState(seed)
        if self.n_splits is None:
            self.n_splits = n_splits

        self.k_fold = k

        if self.mode is None:
            self.mode = SplitType.Random

        self.sample_complexity = sample_complexity
        self.split_idx = None
        self.create_split(data.train.shape, data.train)

        self.data_dim = data.train.shape[1]

        self.options = {AggregationType.Mean: self.dummy,
                        AggregationType.MaximumLikelihood: self.dummy,
                        AggregationType.TukeyDepth: self.dummy,
                        AggregationType.WassersteinBarycenter: self.dummy
                        }

    def dummy(self):
        return None

    def set_mode(self, mode=SplitType.Random):
        """
                Set Split Mode according to existing Enum Members of src.conf.modes.Splits
        """
        self.mode = mode

    def create_split(self, shape, data=None):
        """
            Parameters
            ----------
            shape : Tuple(Integer, Integer)
                Tuple-Object representing the size of the data set usually (rows, cols)

            data : :class:`pandas.DataFrame`
                Either a DataFrame or Numpy Array containing the training data. The data parameter is necessary if a mode is chosen, that
                depends on the data to generate a proper split.

            Notes
            -----
                Creates a Split as collection of numpy arrays, which can be used to index an existing data frame of given shape and size.
            """
        if data is not None:
            try:
                shape = data.shape
            except AttributeError:
                logging.error(
                    "Unable to create split, data was not provided in a valid format. Please use either Pandas Dataframes or Numpy Arrays")

        if self.mode == SplitType.Random:
            self._random(shape)
        elif self.mode == SplitType.Bootstrap:
            if data is None:
                logging.info("Falling back to random split. Data needs to be provided for a data-based Split")
                self._random(shape)
            else:
                self._bootstrap(data)
        else:
            logging.info("Invalid or Unknown Split Type. Falling back to random split")
            self._random(shape)

    def split(self):
        yield from self.split_idx

    def _random(self, shape):
        split = np.arange(shape[0])
        self.random_state.shuffle(split)
        cv_splits = np.array_split(split, self.k_fold)
        self.split_idx = [np.array_split(cv_split, self.n_splits) for cv_split in cv_splits]

    def _bootstrap(self, data):
        total_samples = self.sample_complexity * self.n_splits
        data_idx = np.arange(data.shape[0])
        choices = self.random_state.choice(a=data_idx , size=total_samples, replace=True).reshape(self.n_split, self.sample_complexity)
        self.split_idx = choices

    def _model(self, models):
        local_dist_samples = []
        for model in models:
            mu = np.mean(model, axis=0)
            cov = np.cov(model)
            samples = self.random_state.multivariate_normal(mu, cov, self.sample_complexity)
            local_dist_samples.append(samples)

    def _split_corr(self, data):
        pass


class Random(Sampler):

    def __init__(self, data, n_splits=10, sample_complexity=100, seed=None, k=10):
        self.n_splits = n_splits
        self.mode = SplitType.Random
        super(Random, self).__init__(data, n_splits, sample_complexity, seed, k)


class Bootstrap(Sampler):

    def __init__(self, data, n_splits=10, sample_complexity=100, seed=None, k=10):
        self.n_splits = n_splits
        self.mode = SplitType.Bootstrap
        super(Bootstrap, self).__init__(data, n_splits, sample_complexity, seed, k)


class Model(Sampler):

    def __init__(self, data, n_splits=10, sample_complexity=100, seed=None, k=10):
        self.n_splits = n_splits
        self.mode = SplitType.Model

        super(Model, self).__init__(data, n_splits, sample_complexity, seed, k)
