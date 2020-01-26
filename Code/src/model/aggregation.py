from enum import Enum

import numpy as np



def aggregate(opt, **kwargs):
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
        options[opt](**kwargs)
    except KeyError:
        print("The provided option is unknown defaulting to averaging with uninformative prior")
        options[AggregationType.Mean](**kwargs)


def average(weights):
    """
    Model aggregation by averaging with uninformative prior p=1/n, where n is the number of weight vectors provided.
    Parameters
    ----------
    weights :

    Returns
    -------

    """
    return 1 / weights.shape(0) * np.sum(weights, axis=0)


def weighted_average(weights, distribution):
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
    p = maximum_likelihood(weights, distribution)
    partition = np.zeros(weights.shape[0])
    for i in range(weights.shape[0]):
        partition[i] = p.predict(weights[i])
    return 1 / np.sum(partition) * (weights * partition)


def radon_machine(weights, radon_number, h):
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

    # Coefficient Matrix Ax = b
    r = radon_number
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
            new_weights= _radon_point(A, b) if new_weights is None else np.vstack((new_weights, _radon_point(A, b)))
        aggregation_weights = np.array(new_weights).T
    return aggregation_weights


def _radon_point(A=None, b=None):
    if A is None and b is None:
        print_help()
        return
    sol= np.linalg.solve(A, b)
    pos = sol >= 0
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


def maximum_likelihood(weights, distribution):
    p = 0
    return p


def print_help(aggtype=None):
    pass


def tukey_depth():
    pass


class AggregationType(Enum):
    Mean = 1
    MaximumLikelihood = 2
    RadonPoints = 3
    WassersteinBarycenter = 4
    GeometricMedian = 5
    TukeyDepth = 6


options = {AggregationType.Mean: average,
           AggregationType.MaximumLikelihood: weighted_average,
           AggregationType.RadonPoints: radon_machine,
           AggregationType.TukeyDepth: tukey_depth,
           AggregationType.WassersteinBarycenter: wasserstein_barycenter
           }



