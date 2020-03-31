import os
import io

import numpy as np
import pxpy as px
import networkx as nx
import time

from copy import deepcopy
from shutil import copyfileobj
from multiprocessing import Process
from multiprocessing import cpu_count

from src.data.dataset import Data
from src.conf.bijective_dict import BijectiveDict
from src.conf.settings import CONFIG
from src.model.util.chow_liu_tree import build_chow_liu_tree

LOG_FORMAT = '%(message)s'
LOG_NAME = 'model_logger'
LOG_FILE_INFO = os.path.join('logs', 'model_output.csv')
LOG_FILE_WARN = os.path.join('logs', 'model_warn.log')
LOG_FILE_ERROR = os.path.join('logs', 'model_err.log')
LOG_FILE_DEBUG = os.path.join('logs', 'model_dbg.log')

from time import sleep

logger = CONFIG.get_logger()


def log_progress(start, update, iter_time, total_models, model_count):
    if update is None and iter_time is not None:
        if iter_time - start > 60:
            update = time.time()
            logger.info("Training Models: " +
                        "{:.2%}".format(float(model_count) / float(total_models)))
    if update is not None and iter_time is not None:
        if iter_time - update > 60:
            update = time.time()
            logger.info("Training Models: " +
                        "{:.2%}".format(float(model_count) / float(total_models)))

    return update, iter_time


class Model:

    def __init__(self, data, weights=None, states=None, statespace=None, path=None, delta=0.5, eps=1e-1):
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
                TODO
            delta : Float
                Probability of two average sufficient statistics having at most distance eps
            eps : Float
                Bound for distance between two average sufficient statistics
            See Also
            --------

            Notes
            -----
            We aim to keep Model and Data separate and as such we incorporate the data as an independent object into the PGM.
            For specific Models we may enforce the data to be an inherited class of the :class:``src.data.dataset.Data`.

            Examples
            --------

        """

        self.model_logger = CONFIG.get_logger(LOG_FORMAT, LOG_NAME, LOG_FILE_INFO,
                                              LOG_FILE_WARN, LOG_FILE_ERROR, LOG_FILE_DEBUG)
        if not isinstance(data, Data):
            raise TypeError("Data has to be an instance of Pandas Dataframe")

        self.data_set = data

        self.weights = None
        self.vertices = None
        self.state_space = None
        self.graph = None

        self.hook_counter = 0
        self.stepsize = 1e-1
        self.maxiter = 10000

        self.edgelist = np.empty(shape=(0, 2), dtype=np.uint64)

        self.state_mapping = BijectiveDict()

        self.global_weights = []

        if weights is not None and isinstance(weights, np.ndarray):
            self.weights = weights
        if states is None:
            self.vertices = self._states_from_data()
        if statespace is None:
            self.state_space = self._statespace_from_data()
        if path is not None:
            self.root_dir = os.path.join(CONFIG.ROOT_DIR, "data", path, "model")
            if not os.path.exists(self.root_dir):
                os.makedirs(self.root_dir)

        self._create_graph()
        self.px_batch_local = {}
        self.px_batch = {}
        self.px_model = []
        self.px_model_scaled = []

        self.trained = False
        self.curr_model = 0

        self.train_counter = 0
        self.delta = delta
        self.eps = eps
        self.sample_func = lambda x: 0.1 * x ** 2
        self.suff_data = int(np.ceil(self.hoefding_bound(self.delta, self.eps)))
        self.data_delta = int(np.ceil(self.suff_data / self.sample_func(5)))
        self.n_local_data = 0

        self.prev_obj = np.infty
        self.best_objs = {}
        self.best_weights = {}

        self.csv_writer = io.StringIO("Model, Objective, n_Data")
        print("Model, Objective, n_Data", file=self.csv_writer)
        print("Eps = " + str(self.eps), file=self.csv_writer)
        print("Delta = " + str(self.delta), file=self.csv_writer)
        print("Data Increment = " + str(self.data_delta), file=self.csv_writer)

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

    def predict(self, px_model=None, epoch=None):
        """
        TODO

        Parameters
        ----------
        px_model : `class:px.Model`
            Model to predict the test data with.

        Raises
        ------
        RuntimeError

        Returns
        -------
            `class:np.array` with predicted test data (all -1 elements are predicted and replaced)
        """
        test = np.ascontiguousarray(self.data_set.test.to_numpy().astype(np.uint16))
        test[:, 0] = -1
        if px_model is None:
            if self.trained:
                return [px_model.predict(test) for px_model in self.px_model]
        else:
            return px_model.predict(test)

    def train(self, epochs=1, iters=100, split=None, n_models=None, mode=px.ModelType.mrf):
        """
        TODO

        Parameters
        ----------
        epochs : int
            Number of outer iterations
        iters: int
            Number of inner iteration (per call of px.train)
        split: Split
            class:src.preprocessing.split.Split Contains the number of splits and thus the number of models to be trained.
            Each split will be distributed to a single device/model.

        Raises
        ------
        RuntimeError

        Returns
        -------
            None
        """
        self.maxiter = iters
        self.train_counter += 1
        models = []
        scaled_models = []
        train = np.ascontiguousarray(self.data_set.train.to_numpy().astype(np.uint16))

        # Timing
        start = time.time()
        update = None
        iter_time = None

        # Initialization for best Params
        if split is None:
            total_models = 1
            split = [np.arange(train.shape[0])]
        else:
            total_models = len(split) if n_models is None else np.min([len(split), n_models])

        self.best_weights[self.train_counter] = [0] * total_models
        self.best_objs[self.train_counter] = [np.infty] * total_models
        self.px_batch_local[self.train_counter] = [0] * total_models
        # Distributed Training
        for i, idx in enumerate(split):
            self.curr_iter = 0
            if len(split) > 1:
                self.n_local_data = np.int64(
                    np.min([np.ceil(self.data_delta * self.sample_func(self.train_counter)), idx.shape[0]]))
            else:
                self.n_local_data = idx.shape[0]
            self.curr_model = i
            if n_models is not None:
                if i >= n_models:
                    break
            update, _ = log_progress(start, update, iter_time, total_models, i)
            data = np.ascontiguousarray(np.copy(train[idx[:self.n_local_data].flatten()]))
            # init_data = np.ascontiguousarray(self.state_space.astype(np.uint16).reshape(self.state_space.shape[0],1)).T
            # data = np.ascontiguousarray(np.vstack((data, init_data)))
            model = px.train(data=data,
                             graph=self.graph,
                             iters=iters,
                             shared_states=False,
                             opt_progress_hook=self.opt_progress_hook,
                             mode=CONFIG.MODELTYPE,
                             initial_stepsize=1e-2,
                             opt_regularization_hook=CONFIG.REGULARIZATION,
                             inference=px.InferenceType.junction_tree,
                             k=4)
            self.prev_obj = np.infty
            if len(split) > 1:
                self.px_batch_local[self.train_counter][i] = model
                model = self.merge_weights(model)
            if self.train_counter == 1:
                models.append(model)
                if CONFIG.MODELTYPE == px.ModelType.integer:
                    scaled_models.append(self.scale_model(model))
            else:
                self.px_model[i] = model
                scaled_model = self.scale_model(model)
                self.px_model_scaled[i] = scaled_model
            iter_time = time.time()

        if not self.px_model:
            self.px_model = models
            if CONFIG.MODELTYPE == px.ModelType.integer:
                self.px_model_scaled = scaled_models
        self.px_batch[self.train_counter] = self.px_model
        end = time.time()
        logger.info("Finished Training Models: " +
                    "{:.2f} s".format(end - start))

        if not self.trained:
            self.trained = True

    def scale_model(self, model):
        weights = np.log(2) * np.ascontiguousarray(np.copy(model.weights))
        return px.Model(weights=weights, graph=self.graph, states=self.state_space + 1)

    def merge_weights(self, px_model):
        """
        """

        global_states = np.ascontiguousarray(self.state_space, dtype=np.uint64) + 1
        global_cliques = [global_states[i] * global_states[j] for i, j in self.edgelist]
        model_size = np.sum(global_cliques)

        weights = px_model.weights
        if weights.shape[0] < model_size:
            weights = np.ascontiguousarray(np.copy(px_model.weights))
            local_states = px_model.states
            local_cliques = [local_states[i] * local_states[j] for i, j in self.edgelist]

            missing_states = np.array(global_cliques) - np.array(local_cliques)
            offset = 0
            for j, idx in enumerate(global_cliques):
                if missing_states[j] > 0:
                    inserts = np.zeros(missing_states[j]) + np.min(weights[int(offset):int(offset + local_cliques[j])])
                    weights = np.insert(weights, int(offset + local_cliques[j]), inserts)
                offset += idx
        self.best_weights[self.train_counter][self.curr_model] = weights
        return px.Model(weights=weights.astype(np.float64), graph=px_model.graph, states=global_states)

    def merge_states(self):
        """

        Returns
        -------

        """
        local_states = np.array([model.states for model in self.px_model])
        return np.max(local_states, axis=0)

    def opt_progress_hook(self, state_p):
        """

        Parameters
        ----------
        state_p :

        Returns
        -------

        """

        contents = state_p.contents
        # self.best_weights[self.train_counter][self.curr_model] = np.copy(contents.best_weights)
        if CONFIG.MODELTYPE != px.ModelType.integer:
            if self.check_convergence(np.copy(contents.obj), np.copy(contents.gradient)):
                # logger.info("Optimization Done after " + str(self.curr_iter) + " Iterations")
                if contents.iteration > 100:
                    state_p.contents.iteration = self.maxiter
        self.prev_obj = contents.obj
        self.best_objs[self.train_counter][self.curr_model] = np.min(
            [self.best_objs[self.train_counter][self.curr_model], np.copy(contents.obj).ravel()])
        print(str(self.curr_model) + "," + str(contents.obj) + "," + str(self.n_local_data), file=self.csv_writer)
        self.curr_iter += 1

    def squared_l2_regularization(self, state_p):
        state = state_p.contents
        lam = 0.1
        np.copyto(state.gradient, state.gradient + 2.0 * lam * state.weights)

    def prox_l1(self, state_p):
        state = state_p.contents
        l = state.lam * state.stepsize

        x = state.weights_extrapolation - state.stepsize * state.gradient

        np.copyto(state.weights, 0, where=np.absolute(x) < l)
        np.copyto(state.weights, x - l, where=x > l)
        np.copyto(state.weights, x + l, where=-x > l)

    def progress_hook(self, state_p):
        return

    def opt_regularization_hook(self, state_p):
        return

    def opt_proximal_hook(self, state_p):
        return

    def check_convergence(self, curr_obj, grad, tol=1e-5, gtol=1e-7):
        if CONFIG.MODELTYPE == px.ModelType.integer:
            return self.prev_obj - curr_obj < tol
        else:
            return self.prev_obj - curr_obj < tol and np.linalg.norm(grad, np.infty) < gtol

    def parallel_train(self, split=None):
        # This is slow and bad, maybe distribute proc   esses among devices.
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

    def write_progress_hook(self, path, fname="stats.csv"):
        with open(os.path.join(path, fname), "w+", encoding='utf-8') as f:
            self.csv_writer.seek(0)
            copyfileobj(self.csv_writer, f)

    def _parallel_train(self, data, model):
        px.train(data=data, iters=100, shared_states=False, in_model=model, mode=CONFIG.MODELTYPE,
                 opt_regularization_hook=CONFIG.REGULARIZATION)

    def _create_graph(self):
        """
            Creates an independency structure (graph) from data. The specified mode for the independency structure is used,
            when creating this object.
        """
        holdout = np.ascontiguousarray(self.data_set.holdout.to_numpy().astype(np.uint16))
        self.edgelist = px.train(data=holdout, graph=px.GraphType.auto_tree, iters=1, mode=CONFIG.MODELTYPE,
                                 opt_regularization_hook=CONFIG.REGULARIZATION).graph.edgelist
        self.graph = self._px_create_graph()
        self.weights = self.init_weights()

    def _px_create_graph(self):
        return px.create_graph(self.edgelist, self.state_space)

    def _statespace_from_data(self):
        """
        Generates an array with the number of states for each feature in the order provided through the data.
        Returns
        -------
        :class:`numpy.ndarray` containing the state space for each feature.
        """
        data = self.data_set.train.append(self.data_set.test)
        statespace = np.arange(data.shape[1], dtype=np.uint64)
        for i, column in enumerate(data.columns):
            self.state_mapping[column] = i
            statespace[i] = np.max(data[column].to_numpy().astype(np.uint64))

        return statespace

    def _states_from_data(self):
        states = len(self.data_set.data.columns)
        return states

    def _px_create_model(self):
        return px.Model(weights=self.weights, graph=self.graph,
                        states=self.state_space.reshape(self.state_space.shape[0], 1),
                        stats=px.StatisticsType.overcomplete)

    def _px_create_dist_models(self):
        pass

    def _gen_chow_liu_tree(self):
        if not self.data_set.features_dropped:
            try:
                graph = nx.read_edgelist(os.path.join(self.root_dir, "chow_liu.graph"))
                return np.array([e for e in graph.edges], dtype=np.uint64)
            except FileNotFoundError as fnf:
                logger.error(str(fnf))
                logger.info("Can not find edgelist in folder, generating new chow liu tree - this may take some time.")
        chow_liu_tree = build_chow_liu_tree(self.data_set.holdout.to_numpy(), len(self.data_set.vertices()))
        nx.write_edgelist(chow_liu_tree, os.path.join(self.root_dir, "chow_liu.graph"))
        nx.write_weighted_edgelist(chow_liu_tree, os.path.join(self.root_dir, "chow_liu_weighted.graph"))
        return np.array([e for e in chow_liu_tree.edges], dtype=np.uint64)

    def init_weights(self):
        suff_stats = 0
        for s, t in self.edgelist:
            suff_stats += self.state_space[s] * self.state_space[t]
        suff_stats = int(suff_stats)
        return np.zeros(suff_stats)

    def get_weights(self):
        return np.stack(self.best_weights[self.train_counter], axis=0).T

    def get_num_of_states(self):
        num_states = self.state_space + 1
        return np.sum([num_states[i] * num_states[j] for i, j in self.edgelist])

    def hoefding_bound(self, delta=0.8, eps=1):
        d = self.get_num_of_states()
        c = - (np.log(1 - np.sqrt(delta)) - np.log(2)) / (np.log(d))
        return (2 * (1 + c) * np.log(d)) / eps ** 2

    def get_bounded_distance(self, delta=0.8):
        d = self.get_num_of_states()
        c = - (np.log(1 - np.sqrt(delta)) - np.log(2)) / (np.log(d))
        return 2 * np.sqrt(((1 + c) * np.log(d)) / (2 * self.n_local_data * self.train_counter))


class Dota2(Model):

    def __init__(self, data, weights=None, states=None, statespace=None, path=None):
        super(Dota2, self).__init__(data, weights, states, statespace, path)

        self.data = data
        if states is None:
            self.states = self._states_from_data()
        if statespace is None:
            self.state_space = self._statespace_from_data()

        self.state_mapping = self._set_state_mapping()

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
                    n_edges += (len(token) * (len(token) - 1)) / 2
                    for vertex in token:
                        clique.append(self.state_mapping[vertex])
                    for i, source in enumerate(clique):
                        for j in range(i, len(clique)):
                            if i != j:
                                edge = np.vstack((edge, np.array([source, clique[j]], dtype=np.uint64).reshape(1, 2)))
            assert edge.shape[0] == n_edges
            self.add_edge(np.array(edge))


class Susy(Model):

    def __init__(self, data, weights=None, states=None, statespace=None, path=None):

        self.data = data

        super(Susy, self).__init__(data, weights, states, statespace, path)

        if states is None:
            self.states = self._states_from_data()
        if statespace is None and self.state_space is None:
            self.state_space = self._statespace_from_data()

    def predict(self, px_model=None, n_test=None):
        test = np.ascontiguousarray(self.data_set.test.to_numpy().astype(np.uint16))
        test[:, self.data_set.label_column] = -1
        if n_test is None:
            n_test = test.shape[0] - 1
        else:
            np.min([n_test, test.shape[0] - 1])
        test = np.ascontiguousarray(test[:n_test])
        if px_model is None:
            if self.trained:
                if CONFIG.MODELTYPE == px.ModelType.integer:
                    return [px_model.predict(np.ascontiguousarray(np.copy(test[:n_test]))) for px_model in
                            self.px_model_scaled]
                else:
                    return [px_model.predict(np.ascontiguousarray(np.copy(test[:n_test]))) for px_model in
                            self.px_model]
        else:
            return px_model.predict(test[:n_test])
