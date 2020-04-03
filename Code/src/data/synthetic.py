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
            random_sample = self.dist(self.mean[i], self.cov[i], self.n_samples)
            labels = (np.zeros(self.n_samples) + i).reshape(self.n_samples, 1)
            subclass = np.hstack((labels, random_sample))
            data.append(subclass)
        return data


def main():
    n_dim = 5
    n_classes = 2
    mean = [np.random.rand(n_dim) * np.random.randint(1, 6) for i in range(n_classes)]
    cov =  [np.random.rand(n_dim, n_dim) * np.random.randint(1, 4) for i in range(n_classes)]
    gen =  DataGenerator(5000, 2, np.random.multivariate_normal, mean, cov)
    return gen.sample()


if __name__ == '__main__':
    data = main()

    res = None
    for arr in data:
        res = arr if res is None else np.vstack((res, arr))

    res = np.ascontiguousarray(res, dtype=np.float64)
    disc, M = px.discretize(res, 10)
    model = px.train(disc, graph=px.GraphType.auto_tree)
    fi = fisher_information(model)
    phis = []
    for d in disc:
        phis.append(model.phi(d))
    mu, a = model.infer()
    mu = mu[:vars]
    vars = model.weights.shape[0]
    cov_XY = np.cov(np.array(phis).T)
    EX_EY = np.outer(mu, mu)
    E_XY = cov_XY + EX_EY
    new_data = os.path.join(CONFIG.ROOT_DIR, "data")
    os.chdir(new_data)
    os.mkdir("SYNTH")
    df = pd.DataFrame(disc)
    df.to_csv(os.path.join("SYNTH", "data.csv"))