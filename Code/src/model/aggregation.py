import numpy as np

from src.conf.modes.aggregation_type import AggregationType


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

    Compute the System of Linear Equations SUM_i^r lambda_i s_i = 0     s.t. SUM_i^r lambda_i = 0
    Parameters
    ----------
    weights :
    radon_number :

    Returns
    -------

    """

    # Coefficient Matrix Ax = b
    A = np.array(weights, dtype=np.float64)

    b = np.zeros(radon_number)
    b[b.shape[0] - 1] = 1

    sum_zero_constraint = np.ones(radon_number)
    fix_variable_constraint = np.zeros(radon_number)
    fix_variable_constraint[0] = 1

    A = np.vstack(np.vstack(A, sum_zero_constraint), fix_variable_constraint)


def _radon_point(A=None, b=None):
    if A is None and b is None:
        print_help()
        return


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


options = {AggregationType.Mean: average,
           AggregationType.MaximumLikelihood: weighted_average,
           AggregationType.RadonPoints: radon_machine,
           AggregationType.TukeyDepth: tukey_depth(),
           AggregationType.WassersteinBarycenter: wasserstein_barycenter()
           }
