from enum import Enum

import numpy as np

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

    def __init__(self, model, k=1):
        self.options = {AggregationType.Mean: Mean,
                        AggregationType.MaximumLikelihood: MaximumLikelihood,
                        AggregationType.RadonPoints: RadonMachine,
                        AggregationType.TukeyDepth: tukey_depth,
                        AggregationType.WassersteinBarycenter: wasserstein_barycenter}

        self.model = model
        self.aggregate_models = []
        self.k = k

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

    def __init__(self, model, k=10):
        super(Mean, self).__init__(model, k)

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
        weights = np.array_split(self.model.get_weights(), self.k, axis=1)
        for model in weights:
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

    def __init__(self, model, k, radon_number, h):

        super(RadonMachine, self).__init__(model, k=k)

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
        for i in range(1, self.k):
            folds.append(weights[:, (i-1)*r**h:i*r**h])
        for aggregation_weights in folds:
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



