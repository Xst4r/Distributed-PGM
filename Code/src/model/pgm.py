import os

import pandas as pd
import numpy as np
import networkx as nx

from src.data.dataset import Data
from src.util.bijective_dict import BijectiveDict
from src.util.chow_liu_tree import build_chow_liu_tree
from src.util.conf import ROOT_DIR


class PGM:

    def __init__(self, data, weights=None, states=None, statespace=None, path=None):
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

        self.data_set = data

        self.weights = None
        self.vertices = None
        self.state_space = None

        self.edgelist = np.empty(shape=(0, 2), dtype=np.uint64)

        self.state_mapping = BijectiveDict

        if weights is not None and isinstance(weights, np.ndarray):
            self.weights = weights
        if states is None:
            self.vertices = self._states_from_data()
        if statespace is None:
            self.state_space = self._statespace_from_data()
        if path is not None:
            self.root_dir = os.path.join(ROOT_DIR, "data", path, "model")
            if not os.path.exists(self.root_dir):
                os.makedirs(self.root_dir)

    def _statespace_from_data(self):
        statespace = np.arange(self.data_set.shape[0], dtype=np.uint64)
        for i, column in enumerate(self.data_set.columns):
            self.state_mapping[column] = i
            statespace[i] = np.unique(self.data_set.data[column]).shape[0]

        return statespace

    def _states_from_data(self):
        states = len(self.data_set.data.columns)
        return states

    def get_node_id(self, colname):
        pass

    def get_node_name(self, colid):
        pass

    def add_edge(self, edges):
        if edges.shape[1] != 2:
            raise AttributeError("Invalid Edgelist: Edgelist has to be a 2-Column Matrix of Type np.array")
        self.edgelist = np.vstack((self.edgelist, edges))

    def gen_chow_liu_tree(self):
        try:
            graph = nx.read_edgelist(os.path.join(self.root_dir, "chow_liu.graph"))
            self.chow_liu_tree = np.array([e for e in graph.edges], dtype=np.uint64)
            return
        except FileNotFoundError as fnf:
            print(str(fnf))
            print("Can not find edgelist in folder, generating new chow liu tree - this may take some time.")
        chow_liu_tree = build_chow_liu_tree(self.data.train.to_numpy(), self.vertices)
        nx.write_edgelist(chow_liu_tree, os.path.join(self.root_dir, "chow_liu.graph"))
        nx.write_weighted_edgelist(chow_liu_tree, os.path.join(self.root_dir, "chow_liu_weighted.graph"))
        return chow_liu_tree.edges