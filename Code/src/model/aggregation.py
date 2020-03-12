import os
from enum import Enum
from functools import partial

import numpy as np
import pxpy as px
import warnings
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


from src.conf.settings import get_logger, ROOT_DIR
from src.model.model import Model

logger = get_logger()


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
        if not isinstance(model, (np.ndarray, Model)):
            raise TypeError("Excepted models to be either of type numpy.ndarray, pxpy.model or Model")
        elif isinstance(model, list):
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
            self.aggregate_models.append(self._weighted_average())
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
        if isinstance(self.model, np.ndarray):
            weights = self.model
        else:
            weights = self.model.get_weights()
        distribution = "normal"
        mean, cov = self.estimate_normal(weights)
        try:
            likelihood = multivariate_normal.pdf(weights.T, mean, cov)
        except np.linalg.LinAlgError as e:
            logger.info("Trying to Generate additional samples via Bootstrapping to obtain non-singular matrix")

            alt_cov = self.generate_cov(weights.shape[0])
            bootstrap = np.random.multivariate_normal(mean, alt_cov, weights.shape[1]*2)
            samples = np.hstack((weights, bootstrap.T))
            inverse_alt_cov = np.linalg.inv(np.cov(bootstrap.T))
            print(e)
        try:
             likelihood = multivariate_normal.logpdf(weights.T, mean, alt_cov)
        except Exception as e:
            likelihood = [self.log_normal(x, mean, alt_cov, inverse_alt_cov) for x in weights.T]
        normalizer = np.sum(likelihood)
        return np.sum(likelihood/normalizer * weights, axis=1)

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
            self.aggregate_models.append(res)
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

        if weights.shape[1] != self.radon_number:
            return np.zeros(weights.shape[0])
        # Coefficient Matrix Ax = b
        print("Calculating Radon Point for Radon Number: " + str(self.radon_number) + "\n" +
              "For Matrix with Shape: " + str(weights.shape) + "\n" +
              "using " + str(self.h) + "aggregation layers.")
        r = self.radon_number
        h = self.h
        folds = []
        res = []
        aggregation_weights = weights[:,:r**h]
        for i in range(h, 0, -1):
            new_weights = None
            if i > 1:
                splits = np.split(aggregation_weights, r**(i-1), axis=1)
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
                new_weights= self._radon_point(A, b) if new_weights is None \
                    else np.vstack((new_weights, self._radon_point(A, b)))
            aggregation_weights = np.array(new_weights).T
            res.append(aggregation_weights)
        self.success = True
        return res

    def _radon_point(self, A=None, b=None, sol=None):
        if A is None and b is None and sol is None:
            print_help()
            return

        if sol is None:
            try:
                sol = np.linalg.solve(A, b)
                np.save("leq_sol", sol)
                pos = sol >= 0
            except (ValueError, np.linalg.LinAlgError) as e:
                print(e)
                logger.warn("MATRIX IS SINGULAR")
                sol, resid, rank, s = np.linalg.lstsq(A, b, rcond=None)
                np.save("lstsq_sol", sol)
                pos = sol >= 0

        else:
            pos = sol >= 0

        np.save("coefs", A)
        residue = np.sum(sol[pos]) + np.sum(sol[~pos])
        logger.info("Residue is :" + str(residue))
        normalization_constant = np.sum(sol[pos]) # Lambda
        radon_point = np.sum(sol[pos]/normalization_constant * A[:-2:,pos], axis=1)

        return radon_point


class KL(Aggregation):

    def __init__(self, models, n=100, samples=None, graph=None, states=None, eps=1e-3):
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

        if not all(isinstance(x, px.Model) for x in models) or isinstance(models, np.ndarray):
            raise TypeError("Models have to be either a list of pxpy models or a numpy ndarray containing weights")

        if samples is not None:
            self.X = samples
        else:
            self.X = [model.sample(num_samples=n) for model in models]

        if isinstance(self.model, np.ndarray):
            if graph is None or states is None:
                raise ValueError("Graph and States must be supplied.")
            if isinstance(graph, np.ndarray):
                if graph.shape[1] != 2:
                    raise ValueError("Provided Edgelist has to have exactly 2 Columns")
                self.graph = px.create_graph(graph)
            else:
                self.graph = graph
            self.states = states
            self.weights = self.model
            self.model = [px.Model(weights=weights, graph=graph, states=states) for weights in self.model.T]
        else:
            self.graph = self.model[0].graph
            self.states = self.model[0].states

        self.K = len(self.model)
        self.phi = [model.phi for model in self.model]
        self.obj = np.infty
        self.eps = eps

    def aggregate(self, opt, **kwargs):
        try:
            res = self._aggregate(opt, **kwargs)
            self.success = True
            self.aggregate_models.append(res)
            return res
        except Exception as e:
            logger.error("Aggregation Failed in " + self.__class__.__name__ + " due to " + str(e))

    def _aggregate(self, opt, **kwargs):
        res = np.zeros(self.model[0].weights.shape[0])
        K = self.K
        X = self.X
        average_statistics = []
        for i, samples in enumerate(X):
            avg = np.mean([self.phi[i](x) for x in samples], axis=0)
            average_statistics.append(avg)
        self.average_suff_stats = average_statistics
        x0 = np.zeros(self.model[0].weights.shape[0])
        obj = partial(self.naive_kl, average_statistics=average_statistics,
                                     graph=self.model[0].graph,
                                     states= np.copy(self.model[0].states))
        res = minimize(obj, x0, callback=self.callback, tol=self.eps, options={"maxiter":50})
        kl_model = px.Model(weights=res.x, graph=self.model[0].graph, states=self.model[0].states)
        kl_m, kl_A = kl_model.infer()
        fisher_matrix = []
        for i in range(K):
            self.fisher_information(i, kl_A, res.x)
        return res

    def naive_kl(self, theta, average_statistics, graph, states):
        model = px.Model(weights=theta, graph=graph, states=states)
        avg_stats = np.mean(average_statistics, axis=0)
        _, A = model.infer()
        return -(np.inner(theta, np.mean(average_statistics, axis=0)) - A) + self.l2_regularization(theta)

    def l1_regularization(self, theta, lam=1e-1):
        return lam * np.sum(np.abs(theta))

    def l2_regularization(self, theta, lam=1e-1):
        return  lam * np.sum(np.power(theta, 2))

    def callback(self, theta):
        model = px.Model(weights=theta, graph=self.graph, states=self.states)
        _, A = model.infer()
        obj = -(np.inner(theta, np.mean(self.average_suff_stats, axis=0)) - A)
        print("OBJ: " + str(obj))
        print("DELTA:" + str(np.abs(self.obj - obj)))
        if np.abs(self.obj - obj) < self.eps:
            self.obj = np.nanmin([obj,  self.obj])
            warnings.warn("Terminating optimization: time limit reached")
            return True
        else:
            self.obj = np.nanmin([obj, self.obj])
            return False

    def fisher_information(self,i, kl_A, theta):
        return 1/len(self.X[i]) * \
               np.sum(self.phi[i](x) *
                      np.exp(np.inner(self.phi[i](x), theta) - kl_A) for x in self.X[i])

    def weighted_kl(self):
        pass


class Variance(Aggregation):

    def __init__(self, model, samples, label, graph=None, states=None, edgelist=None):
        super(Variance, self).__init__(model)

        self.edgelist = []
        self.local_data = []
        self.y_true = []
        self.graph = graph
        self.states = states
        if isinstance(self.model, np.ndarray):
            if graph is None or states is None:
                raise ValueError("Models were provided as Collection of weight vectors. "
                                 "Graph or States were None, but need to be specified.")
            if isinstance(graph, np.ndarray):
                self.graph = px.create_graph(edgelist=graph)
            self.weights = self.model
            self.model = [px.Model(weights=weights, graph=self.graph, states=self.states) for weights in self.model.T]
        else:
            self.graph = self.model[0].graph
            self.states = self.model[0].states
        if edgelist is None:
            self.edgelist = self._chain_graph(self.model.shape[1])
        for sample in samples:
            self.y_true.append(sample[:,label])
            sample[:,label] = -1
            self.local_data.append(np.ascontiguousarray(sample, dtype=np.uint16))

    def aggregate(self, opt, **kwargs):
        try:
            res = self._aggregate(opt, **kwargs)
            self.success = True
            self.aggregate_models.append(res)
            return res
        except Exception as e:
            logger.error("Aggregation Failed in " + self.__class__.__name__ + " due to " + str(e))

    def _aggregate(self, opt, **kwargs):
        res = np.zeros(self.model[0].shape[0])
        scores = self._welford()
        return res

    def _chain_graph(self, i):
        return np.append(np.arange(1, i), [0])

    def _welford(self):
        """
        Welford Algorithm for online variance
        :param count:
        :param mean:
        :return:
        """
        def get_acc(y_true, y_pred):
            return np.where(y_true == y_pred)[0].shape[0]/y_true.shape[0]

        def update(model, count, mean, M2, node, y_true):
            count += 1
            remote_data = self.local_data[node]
            predictions = model.predict(remote_data)
            acc = get_acc(y_true, predictions[:,-1])

            delta = acc - mean
            mean += delta/count
            delta2 = acc - mean
            M2 += delta * delta2
            return count, mean, M2

        def finalize(count, mean, M2):
            mean, variance = (mean, M2/count)
            return mean, variance

        scores = []
        for i, model in enumerate(self.model):
            data = self.local_data[i]
            y_true = self.y_true[i]
            y_pred = model.predict(data)[:,-1]
            mean = get_acc(y_true, y_pred)
            count = 1
            M2 = 0
            for node in [self.edgelist[i]]:
                count, mean, M2 = update(self.model[i], count, mean, M2, node, y_true)
            mean, variance = finalize(count, mean, M2)
            scores.append((mean, variance))
        return scores


def wasserstein_barycenter(weights):
    """
    Model aggregation with Wasserstein Barycenters using the Wasserstein-2 Distance, i.e., distance between two discrete probability distributions.
    Parameters
    ----------
    weights :

    Returns
    -------

    """

    pass




def print_help(aggtype=None):
    pass


def kl():
    path = os.path.join(ROOT_DIR, "experiments", "COVERTYPE", "100_50_1_1583660364", "models")

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
    agg = KL(models, 100)
    agg.aggregate(None)

    # KL(theta1 |theta2) = A2 - A1  - mu1(theta2 - theta1)
    # KL(i,j) = A[j] - A[i] - suff_stats[i]*(theta[j] - theta[i])


if __name__ == '__main__':
    kl()


