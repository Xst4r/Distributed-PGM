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
        self.rounds = epochs
        self.n_models = n_models
        self.save_path = os.path.join("experiments", self.name)
        self.seed = None

    def baseline(self, path):
        models = None

        if os.path.isabs(self.name):
            data = self.data_obj(path=self.name, seed=self.seed)
        else:
            data = self.data_obj(path=os.path.join("data", self.name), seed=self.seed)

        if path is None or models is None:
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
                if not os.path.isdir(os.path.join(path, 'baseline')):
                    os.makedirs(os.path.join(path, 'baseline'))
                model.px_model[0].save(os.path.join(path, 'baseline', 'px_model' + str(i)))
                model.write_progress_hook(os.path.join(path, 'baseline'))
                models.append(model)
        else:
            model = self.load_model(path)


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
        d, r, h, n = data.radon_number(r=distributed_models.get_num_of_states() + 2,
                                       h=1,
                                       d=data.train.shape[0])
        test_size = np.min([n_test, data.test.shape[0] - 1])
        methods = ['mean', 'radon', 'wa']
        aggregates = {k: [] for k in methods}
        aggr = [Mean(distributed_models),
                RadonMachine(distributed_models, r, h),
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

                # TODO: Generate new Model Parameters here
                # Aggregation
                logger.info("Aggregating Model No. " + str(i))
                aggregates.append(self.aggregate(trained_model, data))
        return models, aggregates, sampler

    def prepare_and_run(self):

        model_loader = None
        mask = None
        if self.exp_loader is not None:
            experiment_path = os.path.join(self.save_path, "_".join([str(self.num_local_samples), str(self.iters), str(self.rounds)]) + "_" + str(self.exp_loader))
            self.seed = self.load_seed(experiment_path)
            model_loader, mask = self.load_experiment(experiment_path)
        else:
            experiment_path, self.seed = self.save_seed(os.path.join(self.save_path, "_".join([str(self.num_local_samples), str(self.iters), str(self.rounds)])))

        if not os.path.isdir(experiment_path):
            os.makedirs(experiment_path)

        self.baseline(experiment_path)
        data = self.data_obj(path=os.path.join("data", self.name), mask=mask, seed=self.seed)
        models, aggregate, sampler = self.run(data, experiment_path, model_loader)

        if self.exp_loader is None:
            np.save(os.path.join(experiment_path, 'mask'), data.mask)
            np.save(os.path.join(experiment_path, 'splits'), sampler.split_idx)
            os.makedirs(os.path.join(experiment_path, "models"))
            for i, m in enumerate(models):
                np.save(os.path.join(experiment_path, str(i) + "_mask"), m.data.masks[i])
                for j, pxm in enumerate(m.px_model):
                    pxm.save(os.path.join(experiment_path, "models", "k" + str(i) + "_n" + str(j) + "px"))
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
