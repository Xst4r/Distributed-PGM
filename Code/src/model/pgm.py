import pandas as pd
import numpy as np


from src.data.dataset import Data

class PGM:

    def __init__(self, data, weights=None, states=None, statespace=None):
        """
            Parameters
            ----------
            data : :class:`pandas.DataFrame`
                Model weights
            weights : :class:`numpy.ndarray`
                Model weights
            states : Integer
                Undirected graph, representing the conditional independence structure
            statespace : Integer or 1-dimensional :class:`numpy.ndarray`

            See Also
            --------

            Notes
            -----
            We aim to keep Model and Data separate and as such we incorporate the data as an independent object into the PGM.
            For specific Models we may enforce the Data to be a certain child class of Data.

            Examples
            --------

        """
        if not isinstance(data, Data):
            raise TypeError("Data has to be an instance of Pandas Dataframe")

        self.data = data

        self.weights = None
        self.states = None
        self.state_space = None

        if weights is not None and isinstance(weights, np.ndarray):
            self.weights = weights
        if states is None:
            self.states = self._states_from_data()
        if statespace is None:
            self.state_space = self._statespace_from_data()

    def _statespace_from_data(self):
        statespace = np.arange(self.data.shape[0])
        for i, column in enumerate(self.data.columns):
            statespace[i] = np.unique(column)

        return statespace

    def _states_from_data(self):
        states = len(self.data.columns)
        return states

