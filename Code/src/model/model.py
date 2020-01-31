import os

import numpy as np
import pxpy as px
import networkx as nx
import time

from multiprocessing import Process
from multiprocessing import cpu_count

from src.data.dataset import Data
from src.conf.bijective_dict import BijectiveDict
from src.conf.settings import ROOT_DIR, get_logger
from src.model.util.chow_liu_tree import build_chow_liu_tree

LOG_FORMAT = '%(message)s'
LOG_NAME = 'model_logger'
LOG_FILE_INFO = os.path.join('logs', 'model_output.log')
LOG_FILE_WARN = os.path.join('logs', 'model_warn.log')
LOG_FILE_ERROR = os.path.join('logs', 'model_err.log')
LOG_FILE_DEBUG = os.path.join('logs', 'model_dbg.log')

logger = get_logger()

class Model:

    def __init__(self, data, weights=None, states=None, statespace=None, path=None):
        """
            Parameters
            ----------
            data : :class:`data`
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
            For specific Models we may enforce the data to be an inherited class of the :class:``src.data.dataset.Data`.

            Examples
            --------

        """

        self.model_logger = get_logger(LOG_FORMAT, LOG_NAME, LOG_FILE_INFO,
                                       LOG_FILE_WARN, LOG_FILE_ERROR, LOG_FILE_DEBUG)
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
        """
        Adds an edge to the class internal edgelist. Array provided has to contain at least one edge as numpy.ndarray with shape (,2)
        The edgelist is usually of shape (n,2), where n is the total number of edges in the graph.
        Parameters
        ----------
        edges : :class:`numpy.ndarray`
            Array of edges to append to edgelist.
        """
        if edges.shape[1] != 2:
            raise AttributeError("Invalid Edgelist: Edgelist has to be a 2-Column Matrix of Type np.array")
        self.edgelist = np.vstack((self.edgelist, edges))

    def train(self, iter=100, split=None):
        """

        Parameters
        ----------
        iter :
        """
        models  = []
        train = np.ascontiguousarray(self.data_set.train.to_numpy().astype(np.uint16))

        # Timing
        start = time.time()
        update = None
        iter_time = None
        total_models = len(split.split_idx)

        for i, idx in enumerate(split.split()):
            if update is None and iter_time is not None:
                if iter_time - start > 60:
                    update = time.time()
                    logger.info("Training Models: "+
                                "{:.2%}".format(float(i)/float(total_models)))
            if update is not None and iter_time is not None:
                if iter_time - update > 60:
                    update = time.time()
                    logger.info("Training Models: " +
                                "{:.2%}".format(float(i) / float(total_models)))

            data = np.ascontiguousarray(train[idx.flatten()])
            states = np.ascontiguousarray(np.array(self.state_space, copy=True))
            model = px.Model(np.ascontiguousarray(self.init_weights()), self.graph, states=states)
            models.append(model)
            px.train(data=data, iters=iter, shared_states=False, in_model=model)
            iter_time = time.time()
        self.px_model = models
        end = time.time()

        logger.info("Finished Training Models: " +
                    "{:.2f} s".format(end - start))

    def parallel_train(self, split=None):
        # This is slow and bad, maybe distribute processes among devices.
        models = []
        processes = []
        train = np.ascontiguousarray(self.data_set.train.to_numpy().astype(np.uint16))
        states = np.ascontiguousarray(np.array(self.state_space, copy=True))
        weights = np.ascontiguousarray(self.init_weights())
        for i in range(len(split.split_idx)):
            model = px.Model(weights, self.graph, states=states)
            models.append(model)

        for model, idx in zip(models, split.split()):
            data = np.ascontiguousarray(train[idx.flatten()])
            p = Process(target=self._parallel_train, args=(data, model))
            processes.append(p)

        count = 0
        n_proc = cpu_count() - 2
        while count < len(processes):
            if count == len(processes):
                break
            for i in range(count, n_proc):
                if i < len(processes):
                    processes[i].start()

            for i in range(count, n_proc):
                if i < len(processes):
                    processes[i].join()
                    logger.info("Training Models: " +
                            "{:.2%}".format(float(count) / float(len(processes))))

            count += n_proc

        self.px_model = models

    def _parallel_train(self, data, model):
        px.train(data=data, iters=100, shared_states=False, in_model=model)

    def _create_graph(self):
        """
            Creates an independency structure (graph) from data. The specified mode for the independency structure is used,
            when creating this object.
        """
        self.edgelist = self._gen_chow_liu_tree()
        self.graph = self._px_create_graph()
        self.weights = self.init_weights()

    def _statespace_from_data(self):
        """
        Generates an array with the number of states for each feature in the order provided through the data.
        Returns
        -------
        :class:`numpy.ndarray` containing the state space for each feature.
        """
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


class Dota2(Model):

    def __init__(self, data, weights=None, states=None, statespace=None, path=None):

        self.data = data

        if states is None:
            self.states = self._states_from_data()
        if statespace is None:
            self.state_space = self._statespace_from_data()

        super(Dota2, self).__init__(data, weights, states, statespace, path)

        self.state_mapping = self._set_state_mapping()

    def _states_from_data(self):
        return len(self.data.train.columns)

    def _statespace_from_data(self):
        statespace = np.arange(self.states, dtype=np.uint64)
        for i, column in enumerate(self.data.train.columns):
            statespace[i] = np.unique(self.data.train[column]).shape[0]

        return statespace

    def _set_state_mapping(self):
        state_mapping = BijectiveDict()
        for i, column in enumerate(self.data.train.columns):
            state_mapping[str(column)] = i

        return state_mapping

    def edges_from_file(self, path):
        with open(path) as edgelist:
            n_edges = 0
            edges = edgelist.read()
            edges = edges.split(']')
            edge = np.empty(shape=(0, 2), dtype=np.uint64)
            for token in edges:
                token = token.strip("[").split()
                if len(token) < 2:
                    pass
                else:
                    clique = []
                    n_edges += (len(token) * (len(token) - 1))/2
                    for vertex in token:
                        clique.append(self.state_mapping[vertex])
                    for i, source in enumerate(clique):
                        for j in range(i, len(clique)):
                            if i != j:
                                edge = np.vstack((edge, np.array([source, clique[j]], dtype=np.uint64).reshape(1,2)))
            assert edge.shape[0] == n_edges
            self.add_edge(np.array(edge))