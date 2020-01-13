import os

import numpy as np
import pxpy as px
import networkx as nx

from src.data.dataset import Data
from src.conf.bijective_dict import BijectiveDict
from src.conf.settings import ROOT_DIR
from src.model.util.chow_liu_tree import build_chow_liu_tree


class Model:

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
        self.graph = None

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

        self._create_graph()
        self.px_model = self._px_create_model()

    def get_node_id(self, colname):
        pass

    def get_node_name(self, colid):
        pass

    def add_edge(self, edges):
        if edges.shape[1] != 2:
            raise AttributeError("Invalid Edgelist: Edgelist has to be a 2-Column Matrix of Type np.array")
        self.edgelist = np.vstack((self.edgelist, edges))

    def train(self, iter=100):
        data = self.data_set.train.to_numpy()[0:1000]
        #data = self.data_set.train.to_numpy()[0:5000]
        self.px_model = px.train(graph=px.GraphType.chain, data=data[:,4:], iters=5, shared_states=False)
        #self.px_model = px.train(data=data, iters=5, in_model=self.px_model)


    def _create_graph(self):
        self.edgelist = self._gen_chow_liu_tree()
        self.graph = self._px_create_graph()
        self.weights = self.init_weights()

    def _statespace_from_data(self):
        statespace = np.arange(self.data_set.shape[0], dtype=np.uint64)
        for i, column in enumerate(self.data_set.columns):
            self.state_mapping[column] = i
            statespace[i] = np.unique(self.data_set.data[column]).shape[0]

        return statespace

    def _states_from_data(self):
        states = len(self.data_set.data.columns)
        return states

    def _distribute(self):
        pass

    def _gather(self):
        pass

    def _aggregate(self):
        pass

    def _px_create_graph(self):
        return px.create_graph(self.edgelist)

    def _px_create_model(self):
        return px.Model(weights=self.weights, graph=self.graph, states=self.state_space.reshape(self.state_space.shape[0],1), stats=px.StatisticsType.overcomplete)

    def _px_create_dist_models(self):
        pass

    def _gen_chow_liu_tree(self):
        try:
            graph = nx.read_edgelist(os.path.join(self.root_dir, "chow_liu.graph"))
            return np.array([e for e in graph.edges], dtype=np.uint64)
        except FileNotFoundError as fnf:
            print(str(fnf))
            print("Can not find edgelist in folder, generating new chow liu tree - this may take some time.")
        chow_liu_tree = build_chow_liu_tree(self.data_set.train.to_numpy(), len(self.data_set.vertices()))
        nx.write_edgelist(chow_liu_tree, os.path.join(self.root_dir, "chow_liu.graph"))
        nx.write_weighted_edgelist(chow_liu_tree, os.path.join(self.root_dir, "chow_liu_weighted.graph"))
        return chow_liu_tree.edges

    def init_weights(self):
        suff_stats = 0
        for s, t in self.edgelist:
            suff_stats += self.state_space[s] * self.state_space[t]
        suff_stats = int(suff_stats)
        return np.zeros(suff_stats)