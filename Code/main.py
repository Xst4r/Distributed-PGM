from src.data.dataset import Dota2, Susy, CoverType
from src.model.aggregation import RadonMachine, Mean, WeightedAverage
from src.model.model import Dota2 as Dota
from src.model.model import Susy as SusyModel
from src.preprocessing.sampling import Random
from src.conf.settings import get_logger
from time import time
import datetime
import os
from shutil import copyfileobj

import numpy as np
from scipy.stats import random_correlation
import pxpy as px

logger = get_logger()
n_test = 100000


class Coordinator(object):

    def __init__(self, data_set_name, Data, exp_loader, n, k, iters, h, epochs, n_models):
        self.name = data_set_name
        self.data_obj = Data
        self.exp_loader = exp_loader
        self.num_local_samples = n
        self.k_fold = k
        self.iters = iters
        self.h = h
        self.r = 0
        self.rounds = epochs
        self.n_models = n_models
        self.save_path = os.path.join("experiments", self.name)

        self.model_loader = None
        self.mask = None
        if self.exp_loader is not None:
            self.experiment_path = os.path.join(self.save_path, "_".join(
                [str(self.num_local_samples), str(self.iters), str(self.rounds)]) + "_" + str(self.exp_loader))
            self.seed = self.load_seed(self.experiment_path)
            self.model_loader, self.mask = self.load_experiment(self.experiment_path)
        else:
            self.experiment_path, self.seed = self.save_seed(os.path.join(self.save_path, "_".join(
                [str(self.num_local_samples), str(self.iters), str(self.rounds)])))

        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
        self.random_state = np.random.RandomState(seed=self.seed)

    def baseline(self):
        if os.path.isabs(self.name):
            data = self.data_obj(path=self.name, seed=self.seed)
        else:
            data = self.data_obj(path=os.path.join("data", self.name), seed=self.seed)

        models = []
        for i in range(10):
            data.train_test_split(i)
            test_size = np.min([n_test, data.test.shape[0] - 1])
            model = SusyModel(data, path=self.name)
            model.train(split=None, epochs=self.rounds, iters=self.iters)
            predictions = model.predict(n_test=test_size)
            # marginals, A = model.px_model[0].infer()

            accuracy = np.where(np.equal(predictions[0][:, data.label_column], data.test_labels[:test_size]))[0].shape[0] / data.test_labels[:test_size].shape[0]
            print("GLOBAL Model " + str(i) + " : " + str(accuracy))
            if not os.path.isdir(os.path.join(self.experiment_path, 'baseline')):
                os.makedirs(os.path.join(self.experiment_path, 'baseline'))
            model.px_model[0].save(os.path.join(self.experiment_path, 'baseline', 'px_model' + str(i)))
            model.write_progress_hook(os.path.join(self.experiment_path, 'baseline'))
            models.append(model)
        return models

    def load_model(self, path):
        pass

    def save_seed(self, path):
        data_dir = os.path.join(path + "_" + str(int(time())))
        os.makedirs(data_dir)
        seed = np.random.randint(0, np.iinfo(np.int32).max, 1)
        np.save(os.path.join(data_dir, "seed"), seed)
        return data_dir, seed

    def load_experiment(self, path):
        mask = None
        models = [np.load(os.path.join(path, "weights", file)) for file in os.listdir(os.path.join(path, "weights"))]
        try:
            mask = np.load(os.path.join(path, "mask.npy"))
        except FileNotFoundError:
            pass
        return models, mask

    def load_seed(self, path):
        seed = np.load(os.path.join(path, "seed.npy"))
        return seed

    def aggregation_helper(self, model, aggregator, data, weights=None):
        test_size = np.min([n_test, data.test.shape[0] - 1])
        try:
            aggregator.aggregate(None)
            aggregate = aggregator.aggregate_models
            if aggregator.success:
                for aggregate_weights in aggregate:
                    aggregate_model = px.Model(weights=aggregate_weights, graph=model.graph, states=model.state_space + 1)
                    predictions = model.predict(aggregate_model, test_size)
                    accuracy = np.where(np.equal(predictions[:, model.data_set.label_column], data.test_labels[:test_size]))[0].shape[0] / data.test_labels[:test_size].shape[
                        0]
                    logger.info(str(accuracy))
                return {'weights': aggregate_weights, 'labels': predictions, 'acc': accuracy}
            return {'weights': None, 'labels': None, 'acc': None}
        except ValueError or TypeError as e:
            print(e)

    def train(self, data, i, sampler, experiment_path):
        model = SusyModel(data,
                          path=data.__class__.__name__)
        model.train(split=sampler.split_idx,
                    epochs=1,
                    n_models=self.n_models,
                    iters=self.iters)
        model.write_progress_hook(path=experiment_path,
                                  fname=str(i) + ".csv")
        np.save(os.path.join(self.save_path,
                             "weights_" + str(i)),
                model.get_weights())

        return model

    def aggregate(self, distributed_models, data):
        test_size = np.min([n_test, data.test.shape[0] - 1])
        methods = ['mean', 'radon', 'wa']
        aggregates = {k: [] for k in methods}
        aggr = [Mean(distributed_models),
                RadonMachine(distributed_models, self.r, self.h),
                WeightedAverage(distributed_models)]
        weights = None
        if isinstance(distributed_models, np.ndarray):
            weights = distributed_models
        for name, aggregator in zip(methods, aggr):
            aggregates[name].append(
                self.aggregation_helper(model=distributed_models,
                                        aggregator=aggregator,
                                        data=data,
                                        weights=weights))

        return aggregates

    def run(self, data, experiment_path, loaded_model):
        models = []
        aggregates = []
        sampler = None

        if loaded_model is not None:
            models = loaded_model
            dummy_model = SusyModel(data, path="SUSY")
            r = loaded_model[0].shape[0]
            d, r, h, n = data.radon_number(r=r + 2, h=self.h, d=data.train.shape[0])
        else:
            # Outer Cross-Validation Loop.
            for i in range(self.k_fold):
                #Training
                data.train_test_split(i)
                sampler = Random(data, n_splits=data.train.shape[0] / self.num_local_samples, k=self.k_fold, seed=self.seed)
                sampler.create_split(data.train.shape, data.train)
                trained_model = self.train(data=data, i=i, sampler=sampler, experiment_path=experiment_path)
                models.append(trained_model)

                d, r, h, n = data.radon_number(r=trained_model.get_num_of_states() + 2,
                                               h=1,
                                               d=data.train.shape[0])
                self.r = r
                samples = self.sample_parameters(trained_model)
                # TODO: Generate new Model Parameters here
                # Aggregation
                logger.info("Aggregating Model No. " + str(i))
                aggregates.append(self.aggregate(trained_model, data))
        return models, aggregates, sampler

    def sample_parameters(self, model):
        n_samples = self.r ** self.h
        samples_per_model = np.ceil(n_samples/len(model.px_model))
        theta_samples = []
        for px_model in model.px_model:
            cov = self.gen_unif_cov(px_model.weights.shape[0])
            theta_samples.append(self.random_state.multivariate_normal(px_model.weights, cov, samples_per_model))
        return theta_samples

    def gen_unif_cov(self, n_dim):
        return np.diag(np.ones(n_dim))

    def gen_random_cov(self, n_dim):
        eigs = self.random_state.rand(n_dim)
        eigs = eigs/np.sum(eigs) * eigs.shape[0]
        return random_correlation.rvs(eigs, random_state=self.random_state)

    def gen_fisher_cov(self, phi, mu):
        return np.outer(mu-phi, (mu-phi).T)

    def prepare_and_run(self):

        self.baseline()
        data = self.data_obj(path=os.path.join("data", self.name), mask=self.mask, seed=self.seed)
        models, aggregate, sampler = self.run(data, self.experiment_path, self.model_loader)

        if self.exp_loader is None:
            np.save(os.path.join(self.experiment_path, 'mask'), data.mask)
            np.save(os.path.join(self.experiment_path, 'splits'), sampler.split_idx)
            os.makedirs(os.path.join(self.experiment_path, "models"))
            for i, m in enumerate(models):
                np.save(os.path.join(self.experiment_path, str(i) + "_mask"), m.data.masks[i])
                for j, pxm in enumerate(m.px_model):
                    pxm.save(os.path.join(self.experiment_path, "models", "k" + str(i) + "_n" + str(j) + "px"))
        return models, aggregate


def main():
    """
        data = Data(params)
        split = Split(params)

        model = Model(data, split)
        model.train()
        agg = Aggregation()
        agg.aggregate()
    """
    # Create Data and Model
    pass


if __name__ == '__main__':
    loader = 1583334301
    data_set="COVERTYPE"
    data_class = CoverType
    number_of_models = 100
    number_of_cv_splits = 10
    maxiter = 50
    number_of_rounds = 1
    number_of_samples_per_model = 100
    coordinator = Coordinator(data_set_name=data_set,
                              Data=data_class,
                              exp_loader=None,
                              n=number_of_samples_per_model,
                              k=number_of_cv_splits,
                              iters=maxiter,
                              h=1,
                              epochs=number_of_rounds,
                              n_models=number_of_models)

    result, agg = coordinator.prepare_and_run()

    for key, aggregation_method in agg.items():
        print(key)
        for model in aggregation_method:
            print(str(model['acc']))
