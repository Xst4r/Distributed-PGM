import os
from enum import Enum
from functools import partial

import numpy as np
import pxpy as px
import warnings
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

from src.conf.settings import CONFIG
from src.model.model import Model

logger = CONFIG.get_logger()


class AggregationType(Enum):
    Mean = 1
    WeightedAverage = 2
    RadonPoints = 3
    WassersteinBarycenter = 4
    GeometricMedian = 5
    TukeyDepth = 6


class Aggregation:

    def __init__(self, model):
        """

        Parameters
        ----------
        model :
            Either List of `pxpy.Model` or `numpy.ndarray` with shape (row,col),
             where the matrix is a collection of parameter vectors, with row being the number of parameters per model
             and col being the number of models.

            If model is of type `numpy.ndarray`, depending on the Aggregation method a graph and the number of states have
            to be supplied as well. This is usually the case when we need to create an instance of `pxpy.Model` to
            perform inference or sample from the probabilistic graphical model.
        """
        if not isinstance(model, (np.ndarray, Model)) and not isinstance(model, list):
            raise TypeError("Excepted models to be either of type numpy.ndarray, pxpy.model or Model")
        if isinstance(model, list):
            if not all(isinstance(x, px.Model) for x in model):
                raise TypeError("Provided List has to contain objects of type pxpy.Model only")

        self.model = model
        self.aggregate_models = []
        self.success = False

    def aggregate(self, opt, **kwargs):
        """
        Model aggregation by averaging with uninformative prior p=1/n, where n is the number of weight vectors provided.
        Parameters
        ----------
        opt : :class:`AggregationType`
        kwargs: Params to passed to the aggregation function

        Returns
        -------

    """
        raise NotImplementedError("Abstract Function that should not be called.")


class Mean(Aggregation):

    def __init__(self, model):
        super(Mean, self).__init__(model)

        if isinstance(model, Model):
            self.weights = model.get_weights()
        elif isinstance(model, list):
            self.weights = [m.weights for m in self.model]
        else:
            self.weights = self.model

    def aggregate(self, opt, **kwargs):
        try:
            res = self._average()
            if CONFIG.MODELTYPE == px.ModelType.integer:
                res = res * np.log(2)
            self.success = True
            self.aggregate_models.append(res)
            return res
        except Exception as e:
            logger.error("Aggregation Failed in " + self.__class__.__name__ + " due to " + str(e))

    def _average(self):
        """
        Model aggregation by averaging with uninformative prior p=1/n, where n is the number of weight vectors provided.
        Parameters
        ----------
        weights :

        Returns
        -------

        """
        weights = self.weights
        return 1 / weights.shape[1] * np.sum(weights, axis=1)


class WeightedAverage(Aggregation):

    def __init__(self, model):
        super(WeightedAverage, self).__init__(model)

    def aggregate(self, opt, **kwargs):
        try:
            res = self._weighted_average()
            if CONFIG.MODELTYPE == px.ModelType.integer:
                res = res * np.log(2)
            self.aggregate_models.append(res)
            self.success = True
        except Exception as e:
            logger.error("Aggregation Failed in " + self.__class__.__name__ + " due to " + str(e))

    def _weighted_average(self):
        """
        Model aggregation by composing a weighted average, taking into account a model prior. The prior for each weight vector (model) is
        estimated with Maximum Likelihood or a different technique as provided using the provided distribution type.
        Parameters
        ----------
        weights :
        distribution :

        Returns
        -------

        """
        likelihood = None
        if isinstance(self.model, np.ndarray):
            weights = self.model
        else:
            weights = self.model.get_weights()
        distribution = "normal"
        mean, cov = self.estimate_normal(weights)
        try:
            likelihood = multivariate_normal.logpdf(weights.T, mean, cov)
            if all(likelihood == 0):
                raise np.linalg.LinAlgError()
        except np.linalg.LinAlgError as e:
            logger.info("Trying to Generate additional samples via Bootstrapping to obtain non-singular matrix")

            alt_cov = self.generate_cov(weights.shape[0])
            bootstrap = np.random.multivariate_normal(mean, alt_cov, weights.shape[1] * 2)
            samples = np.hstack((weights, bootstrap.T))
            inverse_alt_cov = np.linalg.inv(np.cov(bootstrap.T))
            logger.error(e)
        if likelihood is None:
            try:
                likelihood = multivariate_normal.logpdf(weights.T, mean, alt_cov)
            except Exception as e:
                likelihood = [self.log_normal(x, mean, alt_cov, inverse_alt_cov) for x in weights.T]
        normalizer = np.sum(likelihood)
        return np.sum(likelihood / normalizer * weights, axis=1)

    def estimate_normal(self, weights):
        return np.mean(weights, axis=1), np.cov(weights)

    def generate_cov(self, dim):
        return np.identity(dim)

    def log_normal(self, x, mean, cov, inv):
        """

        Parameters
        ----------
        x :
        mean :
        inv :

        Returns
        -------

        """
        from math import pi
        try:
            return 0.5 * - (np.log((2 * pi) ** 2) + np.log(np.linalg.det(cov))) - 0.5 * np.matmul(
                np.matmul((x - mean).T, inv), x - mean)
        except np.linalg.LinAlgError:
            return multivariate_normal.logpdf(x, mean, cov)


class RadonMachine(Aggregation):

    def __init__(self, model, radon_number, h):

        super(RadonMachine, self).__init__(model)

        self.radon_number = int(radon_number)
        self.h = int(h)

    def aggregate(self, opt, **kwargs):
        try:
            res = self._radon_machine()
            if CONFIG.MODELTYPE == px.ModelType.integer:
                res[0] = res[0] * np.log(2)
            self.aggregate_models = res
            self.success = True
            return res
        except Exception as e:
            logger.error("Aggregation Failed in " + self.__class__.__name__ + " due to " + str(e))

    def _radon_machine(self):
        """
        Model aggregation with Radon Machines i.e. Radon Points. The provided Radon Number is usually (in Euclidean Space)  r = d + 2,
        where d is the weight vector dimension. Hence we require at least r weight vectors to solve the system of linear equations necessary to
        compute the radon point.

        See Tukey Depth or Geometric Median.

        Compute the solution to the system of linear equations SUM_i^r lambda_i s_i = 0     s.t. SUM_i^r lambda_i = 0
        Parameters
        ----------
        weights :
        radon_number :

        Returns
        -------

        """
        if not isinstance(self.model, np.ndarray):
            weights = self.model.get_weights()
        else:
            weights = self.model

        if weights.shape[1] < self.radon_number ** self.h:
            raise np.linalg.LinAlgError("Not enough Models for Radon Aggregation")
        else:
            weights = weights[:, :self.radon_number ** self.h]
        # Coefficient Matrix Ax = b
        logger.info("Calculating Radon Point for Radon Number: " + str(self.radon_number) + "\n" +
                    "For Matrix with Shape: " + str(weights.shape) + "\n" +
                    "using " + str(self.h) + " aggregation layers.")
        r = self.radon_number
        h = self.h
        folds = []
        res = []
        aggregation_weights = weights[:, :r ** h]
        for i in range(h, 0, -1):
            new_weights = None
            if i > 1:
                splits = np.split(aggregation_weights, r ** (i - 1), axis=1)
            else:
                splits = [aggregation_weights]
            for split in splits:
                A = split

                b = np.zeros(split.shape[1])
                b[b.shape[0] - 1] = 1

                sum_zero_constraint = np.ones(split.shape[1])
                fix_variable_constraint = np.zeros(split.shape[1])
                fix_variable_constraint[0] = 1

                A = np.vstack((np.vstack((A, sum_zero_constraint)), fix_variable_constraint))
                new_weights = self._radon_point(A, b) if new_weights is None \
                    else np.vstack((new_weights, self._radon_point(A, b)))
            aggregation_weights = np.array(new_weights).T
            res.append(aggregation_weights)
        self.success = True
        return res

    def _radon_point(self, A=None, b=None, sol=None):
        if A is None and b is None and sol is None:
            return

        if sol is None:
            try:
                sol = np.linalg.solve(A, b)
                np.save("leq_sol", sol)
                pos = sol >= 0
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.error(e)
                logger.warn("MATRIX IS SINGULAR")
                sol, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
                np.save("lstsq_sol", sol)
                pos = sol >= 0

        else:
            pos = sol >= 0

        np.save("coefs", A)
        residue = np.sum(sol[pos]) + np.sum(sol[~pos])
        logger.info("Residue is :" + str(residue))
        normalization_constant = np.sum(sol[pos])  # Lambda
        radon_point = np.sum(sol[pos] / normalization_constant * A[:-2:, pos], axis=1)

        return radon_point


class KL(Aggregation):

    def __init__(self, models, n=100, samples=None, graph=None, states=None, eps=1e-2):
        """

        Parameters
        ----------
        models :
        n :
        samples :
        graph :
        states :
        eps :
        """
        super(KL, self).__init__(models)

        if not (all(isinstance(x, px.Model) for x in models) or isinstance(models, np.ndarray)):
            raise TypeError("Models have to be either a list of pxpy models or a numpy ndarray containing weights")

        if isinstance(self.model, np.ndarray):
            if graph is None or states is None:
                raise ValueError("Graph and States must be supplied.")
            if isinstance(graph, np.ndarray):
                if graph.shape[1] != 2:
                    raise ValueError("Provided Edgelist has to have exactly 2 Columns")
                self.graph = px.create_graph(graph)
            else:
                self.graph = graph
            self.states = np.ascontiguousarray(np.copy(states))
            self.weights = self.model
            self.model = [px.Model(weights=weights, graph=graph, states=states) for weights in self.model.T]
        else:
            self.states = np.ascontiguousarray(np.copy(self.model[0].states))
            self.edgelist = np.ascontiguousarray(np.copy(self.model[0].graph.edgelist))
            self.graph = px.create_graph(self.edgelist)

        if samples is not None:
            self.X = [np.copy(sample) for sample in samples]
        else:
            self.X = [model.sample(num_samples=n, sampler=CONFIG.SAMPLER) for model in self.model]

        self.K = len(self.model)
        self.obj = np.infty
        self.eps = eps

    def aggregate(self, opt, **kwargs):
        try:
            opt = True
            res = self._aggregate(opt, **kwargs)
            self.success = True
            self.aggregate_models.append(res)
            return res
        except Exception as e:
            logger.error("Aggregation Failed in " + self.__class__.__name__ + " due to " + str(e))

    def _aggregate(self, opt, **kwargs):
        naivekl = np.zeros(self.model[0].weights.shape[0])
        K = self.K
        X = self.X
        if opt:
            data = np.ascontiguousarray(np.concatenate(X), dtype=np.uint16)
            data = np.ascontiguousarray(np.vstack((data, self.states-1)).astype(np.uint16))
            model = px.train(data=data, graph=self.graph, iters=0)
            s = np.ctypeslib.as_array(model.empirical_stats, shape=(model.dimension,))
            s -= model.phi((self.states-1).ravel())
            model.num_instances -= 1
            res = px.train(in_model=model, opt_regularization_hook=CONFIG.REGULARIZATION)
            merged = self.merge_weights(res, self.states)
            weights = np.ascontiguousarray(merged.weights)
            states = np.ascontiguousarray(self.states)
            kl_model = px.Model(weights=weights.astype(np.float64), graph=self.graph, states=states)
        """
        else:
            average_statistics = []
            for i, samples in enumerate(X):
                avg = np.mean([self.phi[i](x) for x in samples], axis=0)
                average_statistics.append(avg)
            self.average_suff_stats = average_statistics
            x0 = np.zeros(self.model[0].weights.shape[0])
            obj = partial(self.naive_kl, average_statistics=average_statistics,
                          graph=self.model[0].graph,
                          states=np.copy(self.model[0].states))
            res = minimize(obj, x0, callback=self.callback, tol=self.eps, options={"maxiter": 50, "gtol": 1e-3})
            kl_model = px.Model(weights=res.x, graph=self.model[0].graph, states=self.model[0].states)
        """
        naivekl += np.copy(kl_model.weights)
        # self.test(kl_model)
        """
        try:
            fisher_matrix = []
            inverse_fisher = []
            for i in range(K):
                fisher_matrix.append(self.fisher_information(i, kl_m[:kl_model.weights.shape[0]], kl_model.weights))
                inverse_fisher.append(np.linalg.inv(fisher_matrix[i]))
        except np.linalg.LinAlgError as e:
            pass
        """

        return kl_model.weights

    def merge_weights(self, px_model, states):
        """
        """

        global_states = np.ascontiguousarray(states, dtype=np.uint64)
        global_cliques = [global_states[i] * global_states[j] for i, j in px_model.graph.edgelist]
        model_size = np.sum(global_cliques)

        weights = px_model.weights
        if weights.shape[0] < model_size:
            weights = np.ascontiguousarray(np.copy(px_model.weights))
            local_states = px_model.states
            local_cliques = [local_states[i] * local_states[j] for i, j in px_model.graph.edgelist]

            missing_states = np.array(global_cliques) - np.array(local_cliques)
            offset = 0
            for j, idx in enumerate(global_cliques):
                if missing_states[j] > 0:
                    inserts = np.zeros(missing_states[j])
                    weights = np.insert(weights, int(offset + local_cliques[j]), inserts)
                offset += idx

        return px.Model(weights=weights.astype(np.float64), graph=px_model.graph, states=global_states)

    def test(self, model):
        labels = np.concatenate(self.X)[:, -1]
        data = np.concatenate(self.X)
        data[:, -1] = -1
        preds = model.predict(np.ascontiguousarray(np.copy(data), dtype=np.uint16))
        data[:, -1] = -1
        all_preds = [m.predict(np.ascontiguousarray(np.copy(data), dtype=np.uint16)) for m in self.model]
        acc = np.where(preds[:, -1] == labels)[0].shape[0] / labels.shape[0]
        all_acc = [np.where(p[:, -1] == labels)[0].shape[0] / labels.shape[0] for p in all_preds]

        return all_acc

    def naive_kl(self, theta, average_statistics, graph, states):
        model = px.Model(weights=theta, graph=graph, states=states)
        avg_stats = np.mean(average_statistics, axis=0)
        _, A = model.infer()
        return -(np.inner(theta, np.mean(average_statistics, axis=0)) - A) + self.l1_regularization(theta)

    def l1_regularization(self, theta, lam=1e-2):
        return lam * np.sum(np.abs(theta))

    def l2_regularization(self, theta, lam=0):
        return lam * np.sum(np.power(theta, 2))

    def px_callback(self, state_p):
        pass

    def callback(self, theta):
        model = px.Model(weights=theta, graph=self.graph, states=self.states)
        _, A = model.infer()
        obj = -(np.inner(theta, np.mean(self.average_suff_stats, axis=0)) - A)
        # print("OBJ: " + str(obj))
        # print("REG: " + str(self.l2_regularization(theta)))
        # print("DELTA:" + str(np.abs(self.obj - obj)))
        if np.abs(self.obj - obj) < self.eps:
            self.obj = np.nanmin([obj, self.obj])
            warnings.warn("Terminating optimization: time limit reached")
            return True
        else:
            self.obj = np.nanmin([obj, self.obj])
            return False

    def true_fisher(self, model, states, edgelist):
        mu, A = model.infer()
        state_space = [(states[edge[0]] + 1) * (states[edge[1]] + 1) for edge in edgelist]
        fisher_information = np.zeros((np.sum(state_space), np.sum(state_space)))
        data_vec = np.zeros(len(edgelist) + 1, dtype=np.uint16) - 1
        total = 0
        phi = np.zeros(np.sum(state_space))
        for i in state_space:
            for j in range(i):
                phi[np.sum(state_space[:i]) + j] = 1
                next = self.inverse_phi(phi, np.sum(state_space[:i]), len(edgelist) + 1, edgelist[i],
                                        states[edgelist[i]][0], states[edgelist[i]][1])
                cond_mu, _ = model.infer(next)
                fish_row = mu[np.sum(state_space[:i]) + j] * cond_mu
                fisher_information[np.sum(state_space[:i]) + j:, ] = fish_row
                phi[np.sum(state_space[:i]) + j] = 0

        return fisher_information

    def inverse_phi(self, phi, offset, n, edge, statex, statey):
        x = np.zeros(n, dtype=np.uint16) - 1
        pos = np.where(phi)[0] - offset
        x[edge[0]] = np.floor(pos / statex)
        x[edge[1]] = np.mod(pos, statex)
        return x

    def fisher_information(self, i, mu, theta):
        res = np.zeros((theta.shape[0], theta.shape[0]))
        for x in self.X[i]:
            res += np.outer(-self.phi[i](x) + mu, - self.phi[i](x) + mu)
        return 1 / self.X[i].shape[0] * res

    def weighted_kl(self):
        pass


class Variance(Aggregation):

    def __init__(self, model, samples, label, graph=None, states=None, edgelist=None):
        super(Variance, self).__init__(model)

        self.edgelist = []
        self.local_data = []
        self.y_true = []
        self.states = states
        if isinstance(self.model, np.ndarray):
            if graph is None or states is None:
                raise ValueError("Models were provided as Collection of weight vectors. "
                                 "Graph or States were None, but need to be specified.")
            if isinstance(graph, np.ndarray):
                self.graph = px.create_graph(graph)
            self.weights = self.model
            self.model = [px.Model(weights=weights, graph=self.graph, states=self.states) for weights in self.model.T]
        else:
            self.px_edgelist = np.ascontiguousarray(np.copy(self.model[0].graph.edgelist))
            self.graph = px.create_graph(self.px_edgelist)
            self.states = np.ascontiguousarray(np.copy(self.model[0].states))
            self.weights = np.array([np.copy(mod.weights) for mod in self.model])
        if edgelist is None:
            self.edgelist = self._full_graph(len(self.model))
        for sample in samples:
            self.y_true.append(np.copy(sample[:, label]))
            sample[:, label] = -1
            self.local_data.append(np.ascontiguousarray(np.copy(sample), dtype=np.uint16))

    def aggregate(self, opt, **kwargs):
        try:
            res = self._aggregate(opt, **kwargs)
            if CONFIG.MODELTYPE == px.ModelType.integer:
                res = res * np.log(2)
            self.success = True
            self.aggregate_models.append(res)
            return res
        except Exception as e:
            logger.error("Aggregation Failed in " + self.__class__.__name__ + " due to " + str(e))

    def _aggregate(self, opt, **kwargs):
        res = self._welford()
        return res

    def _chain_graph(self, i):
        return np.append(np.arange(1, i), [0])

    def _full_graph(self, i):
        edges = np.arange(0, i)
        edgelist = np.zeros((i, i - 1))
        for j in range(i):
            edgelist[j:, ] = np.delete(edges, j)
        return edgelist.astype(np.uint16)

    def _welford(self):
        """
        Welford Algorithm for online variance
        :param count:
        :param
        mean:
        :return:
        """

        def get_acc(y_true, y_pred):
            return np.where(y_true == y_pred)[0].shape[0] / y_true.shape[0]

        def update(model, count, mean, M2, node, y_true):
            count += 1
            remote_data = self.local_data[node]
            predictions = model.predict(remote_data)
            acc = get_acc(y_true, predictions[:, -1])

            delta = acc - mean
            mean += delta / count
            delta2 = acc - mean
            M2 += delta * delta2
            return count, mean, M2

        def finalize(count, mean, M2):
            mean, variance = (mean, M2 / count)
            return mean, variance

        scores = []
        for i, model in enumerate(self.model):
            # (print(str(i)))
            data = self.local_data[i]
            y_true = self.y_true[i]
            y_pred = model.predict(data)[:, -1]
            mean = get_acc(y_true, y_pred)
            count = 1
            M2 = 0
            for node in self.edgelist[i]:
                count, mean, M2 = update(self.model[i], count, mean, M2, node, y_true)
            mean, variance = finalize(count, mean, M2)
            scores.append((mean, variance))
        mean_weights = np.array(scores)[:, 0] / np.sum(np.array(scores)[:, 0])
        res = np.matmul(self.weights.T, mean_weights)
        return res

def kl():
    path = os.path.join(CONFIG.ROOT_DIR, "experiments", "COVERTYPE", "100_50_1_1583660364", "models")

    models = []
    theta = []
    logpartitions = []
    suff_stats = []
    for file in os.listdir(path):
        if "k0" in file:
            models.append(px.load_model(os.path.join(path, file)))
    samples = []
    for model in models:
        m, A = model.infer()
        samples.append(model.sample(num_samples=100))
        logpartitions.append(A)
        suff_stats.append(model.statistics)
        theta.append(model.weights)
    weights = np.array([model.weights for model in models]).T
    var_agg = Variance(weights, samples, -1, models[0].graph, models[0].states)
    var_agg.aggregate(None)
    agg = KL(models, 500)
    agg.aggregate(None)

    # KL(theta1 |theta2) = A2 - A1  - mu1(theta2 - theta1)
    # KL(i,j) = A[j] - A[i] - suff_stats[i]*(theta[j] - theta[i])


if __name__ == '__main__':
    kl()
