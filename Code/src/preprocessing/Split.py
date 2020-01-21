import logging
from math import log

import numpy as np

from src.conf.modes.aggregation_type import AggregationType
from src.conf.modes.split_type import SplitType


class Split:

    def __init__(self, data, devices=10):
        """
            Parameters
            ----------
            devices : Integer
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
        self.devices = devices
        self.split_mode = SplitType.Random
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
        self.split_mode = mode

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

        if self.split_mode == SplitType.Random:
            self._split_random(shape)
        elif self.split_mode == SplitType.Correlation:
            if data is None:
                logging.info("Falling back to random split. Data needs to be provided for a data-based Split")
                self._split_random(shape)
            else:
                self._split_corr(data)
        else:
            logging.info("Invalid or Unknown Split Type. Falling back to random split")
            self._split_random(shape)

    def split(self):
        yield from self.split_idx

    def _split_random(self, shape):
        split = np.arange(shape[0])
        np.random.shuffle(split)
        self.split_idx = np.array_split(split, self.devices)

    def _split_corr(self, data):
        pass
