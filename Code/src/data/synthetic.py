import numpy as np
from src.conf.settings import CONFIG
from src.data.metrics import fisher_information
import pandas as pd
import os
import pxpy as px

class DataGenerator(object):

    def __init__(self, n_samples, n_classes, dist, mean, cov):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.dist = dist
        self.mean = mean
        self.cov = cov

    def sample(self):
        data = []
        for i in range(self.n_classes):
            random_sample = self.dist(self.mean[i], np.dot(self.cov[i], self.cov[i].T), self.n_samples)
            labels = (np.zeros(self.n_samples) + i).reshape(self.n_samples, 1)
            subclass = np.hstack((labels, random_sample))
            data.append(subclass)
        return data


def main():
    n_dim = 5
    n_classes = 2
    mean = [np.random.rand(n_dim) * np.random.randint(1, 6) for i in range(n_classes)]
    cov =  [np.random.rand(n_dim, n_dim) * np.random.randint(1, 4) for i in range(n_classes)]
    gen =  DataGenerator(50000, 2, np.random.multivariate_normal, mean, cov)
    return gen.sample()

from scipy.stats import random_correlation


def gen_semi_random_cov(model, eps = 0):
    a = np.insert(np.cumsum([model.states[u] * model.states[v] for u, v in model.graph.edgelist]), 0, 0)
    marginals, A = model.infer()
    eigs = np.random.rand(model.weights.shape[0])
    eigs = eigs / np.sum(eigs) * eigs.shape[0]
    cov = random_correlation.rvs(eigs)
    cov -= -np.outer(marginals[:model.weights.shape[0]], marginals[:model.weights.shape[0]])
    rhs = np.outer(marginals[:model.weights.shape[0]], marginals[:model.weights.shape[0]])
    diag = np.diag(marginals[:model.weights.shape[0]] - marginals[:model.weights.shape[0]]**2)
    for x in range(a.shape[0] - 1):
        cov[a[x]:a[x + 1], a[x]:a[x + 1]] = - rhs[a[x]:a[x + 1], a[x]:a[x + 1]]
    cov -= np.diag(np.diag(cov))
    cov += diag + np.diag(np.full(model.weights.shape[0], eps))

    return cov

if __name__ == '__main__':
    data = main()

    res = None
    for arr in data:
        res = arr if res is None else np.vstack((res, arr))

    res = np.ascontiguousarray(res, dtype=np.float64)
    disc, M = px.discretize(res, 10)
    model = px.train(disc, graph=px.GraphType.auto_tree, iters=10000)
    gen_semi_random_cov(model, 1e-1)
    mu, A = model.infer()
    vars = model.weights.shape[0]
    mu = mu[:vars]
    fi = np.outer(mu - model.statistics, mu - model.statistics)
    phis = []
    for d in disc:
        phis.append(model.phi(d))
    cov_XY = np.cov(np.array(phis).T)
    EX_EY = np.outer(mu, mu)
    E_XY = cov_XY + EX_EY
    new_data = os.path.join(CONFIG.ROOT_DIR, "data")
    os.chdir(new_data)
    os.mkdir("SYNTH")
    df = pd.DataFrame(disc)
    df.to_csv(os.path.join("SYNTH", "data.csv"))