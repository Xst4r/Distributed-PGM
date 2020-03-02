from enum import Enum

import numpy as np
import pxpy as px

from src.conf.settings import get_logger

logger = get_logger()


class AggregationType(Enum):
    Mean = 1
    MaximumLikelihood = 2
    RadonPoints = 3
    WassersteinBarycenter = 4
    GeometricMedian = 5
    TukeyDepth = 6


class Aggregation:

    def __init__(self, model):
        self.options = {AggregationType.Mean: Mean,
                        AggregationType.MaximumLikelihood: MaximumLikelihood,
                        AggregationType.RadonPoints: RadonMachine,
                        AggregationType.TukeyDepth: tukey_depth,
                        AggregationType.WassersteinBarycenter: wasserstein_barycenter}

        self.model = model
        self.aggregate_models = []

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
        try:
            self.options[opt](**kwargs)
        except KeyError:
            print("The provided option is unknown defaulting to averaging with uninformative prior")
            self.options[AggregationType.Mean](**kwargs)


class Mean(Aggregation):

    def __init__(self, model):
        super(Mean, self).__init__(model)

    def aggregate(self, opt, **kwargs):
        self._average()

    def _average(self):
        """
        Model aggregation by averaging with uninformative prior p=1/n, where n is the number of weight vectors provided.
        Parameters
        ----------
        weights :

        Returns
        -------

        """
        weights = self.model.get_weights()
        self.aggregate_models.append(1 / weights.shape[0] * np.sum(weights, axis=0))


class MaximumLikelihood(Aggregation):

    def __init__(self, model):
        super(MaximumLikelihood, self).__init__(model)

    def aggregate(self, opt, **kwargs):
        self._weighted_average()

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

        weights = self.model.get_weights()
        distribution = "normal"
        p = self.maximum_likelihood(weights, distribution)
        partition = np.zeros(weights.shape[0])
        for i in range(weights.shape[0]):
            partition[i] = p.predict(weights[i])
        return 1 / np.sum(partition) * (weights * partition)

    def maximum_likelihood(self, weights, distribution):
        p = 0
        return p


class RadonMachine(Aggregation):

    def __init__(self, model, radon_number, h):

        super(RadonMachine, self).__init__(model)

        self.radon_number = int(radon_number)
        self.h = int(h)

    def aggregate(self, opt, **kwargs):
        self.aggregate_models = self._radon_machine()

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
        # Coefficient Matrix Ax = b
        print("Calculating Radon Point for Radon Number: " + str(self.radon_number) + "\n" +
              "For Matrix with Shape: " + str(weights.shape) + "\n" +
              "using " + str(self.h) + "aggregation layers.")
        r = self.radon_number
        h = self.h
        folds = []
        res = []
        aggregation_weights = weights
        for i in range(h, 0, -1):
            new_weights = None
            if i > 1:
                splits = np.split(aggregation_weights, r, axis=1)
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
            aggregate = np.array(new_weights).T
            res.append(aggregate)
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

    def __init__(self, model, A, X):
        super(KL, self).__init__(model)
        if not all(isinstance(x, px.Model) for x in self.model):
            raise TypeError("Models have to be PX Models for this Aggregaton")

        self.A = A
        self.X = X
        self.K = len(self.model)
        self.phi = [model.phi for model in self.model]

    def aggregate(self, opt, **kwargs):
        self._aggregate(opt, **kwargs)

    def _aggregate(self, opt, **kwargs):
        from scipy.optimize import minimize
        K = self.K
        A = self.A
        X = self.X
        x0 = np.zeros(self.model[0].weights.shape[0])
        res = minimize(self.naive_kl, x0, args=(self.phi, A, X, K))
        kl_model = px.Model(weights=res.x, graph=self.model[0].graph, states=self.model[0].states)
        kl_A, kl_marginals = kl_model.infer()
        fisher_matrix = []
        for i in range(K):
            self.fisher_information(i, kl_A, rex.x)

    def naive_kl(self, theta, phi, A, X, K):
        def inner(theta, phi, A, X):
            p_x = np.exp([theta * phi(x) - A for x in X])
            return np.sum(p_x)
        return - np.sum([inner(theta, phi[k], A[k], X[k]) for k in K])

    def fisher_information(self,i, kl_A, theta):
        return 1/len(self.X[i]) * \
               np.sum(self.phi[i](x) *
                      np.exp(np.inner(self.phi[i](x), theta) - kl_A) for x in self.X[i])

    def weighted_kl(self):
        pass

class Variance(Aggregation):

    def __init__(self, model):
        super(Variance, self).__init__(model)
        self.edgelist = []
        self.local_data = []

    def aggregate(self, opt, **kwargs):
        self._aggregate(opt, **kwargs)

    def _aggregate(self, opt, **kwargs):
        pass

    def welford(self, count, mean):
        """
        Welford Algorithm for online variance
        :param count:
        :param mean:
        :return:
        """
        def get_acc(predictions):
            return 0.5

        def update(count, mean, M2, node):
            theta_new = self.model[node]
            model = px.Model(weights=theta_new, graph=None, states=None)
            predictions = model.predict(self.local_data[node])
            acc = get_acc(predictions)
            new_mean = (acc - mean)/count
            M2 += (acc - mean)*(acc - new_mean)
            return count, mean, M2

        def finalize(count, mean, M2):
            mean, variance = (mean, M2/count)
            return mean, variance

        scores = []
        for i, theta in enumerate(self.model):
            mean = theta
            count = 1
            M2 = theta
            for node in self.edgelist[i]:
                mean, count, M2 = update(count, mean, M2, node)
            mean, variance = finalize(count, mean, M2, node)
            scores.append((mean, variance))


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


def tukey_depth():
    pass



