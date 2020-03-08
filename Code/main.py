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


def baseline(Data, name, iters, epochs, path=None, seed=None):
    models = None

    if os.path.isabs(name):
        data = Data(path=name, seed=seed)
    else:
        data = Data(path=os.path.join("data", name), seed=seed)

    if path is None or models is None:
        models = []
        for i in range(10):
            data.train_test_split(i)
            test_size = np.min([n_test, data.test.shape[0] - 1])
            model = SusyModel(data, path=name)
            model.train(split=None, epochs=epochs, iters=iters)
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
        model = load_model(path)


def load_model(path):
    pass


def save_seed(path):
    data_dir = os.path.join(path + "_" + str(int(time())))
    os.makedirs(data_dir)
    seed = np.random.randint(0, np.iinfo(np.int32).max, 1)
    np.save(os.path.join(data_dir, "seed"), seed)
    return data_dir, seed


def load_experiment(path):
    mask = None
    models = [np.load(os.path.join(path, "weights", file)) for file in os.listdir(os.path.join(path, "weights"))]
    try:
        mask = np.load(os.path.join(path, "mask.npy"))
    except FileNotFoundError:
        pass
    return models, mask


def load_seed(path):
    seed = np.load(os.path.join(path, "seed.npy"))
    return seed


def aggregation_helper(model, aggregator, data, weights=None):
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


def dota():
    data = Dota2(path="data/DOTA2")
    data.drop([23, 'ClusterID', 'GameMode', 'GameType'])
    model = Dota(data, path="DOTA2")

    d, r, h, n = data.radon_number(r=model.weights.shape[0] + 2)
    split = Random(data, n_splits=r ** h)
    print("Weights: " + str(model.weights.shape) + "\n" +
          "Radon Number " + str(r) + "\n")

    # Train Models
    model.train(split=split, epochs=2, iters=20)

    # predictions = model.predict()
    # accuracy = np.where(np.equal(predictions[:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
    # return model, None, predictions, accuracy
    # Radon Machines
    try:
        rm = RadonMachine(model, r, h)
        rm.aggregate(None)
        radon_point = rm.aggregate_models[0]
    except ValueError or TypeError as e:
        print("bla")

    # Create new model with radon point and predict labels
    aggregate_model = px.Model(weights=radon_point, graph=model.graph, states=model.state_space)

    predictions = model.predict(aggregate_model)
    accuracy = np.where(np.equal(predictions[:, data.label_column], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
    return model, radon_point, predictions, accuracy


def prep_and_train(cv_split, data_index, iters, data, save_path, n_models=None):
    model = SusyModel(data,
                      path=data.__class__.__name__)
    state_space = model.state_space + 1
    model.train(split=data_index,
                epochs=1,
                n_models=n_models,
                iters=iters)
    model.write_progress_hook(path=save_path,
                              fname=str(cv_split) + ".csv")
    np.save(os.path.join(save_path,
                         "weights_" + str(cv_split)),
            model.get_weights())
    return model


def run(data, save_path, model=None,
        n=100, k=10, iters=500, h=1, epochs=1, seed=None, n_models=None):
    models = []
    if model is None:

        for i in range(k):
            data.train_test_split(i)
            sampler = Random(data, n_splits=data.train.shape[0] / n, k=k, seed=seed)
            sampler.create_split(data.train.shape, data.train)
            models.append(prep_and_train(cv_split=i,
                                         data_index=sampler.split_idx,
                                         iters=iters,
                                         data=data,
                                         save_path=save_path,
                                         n_models=n_models))

        d, r, h, n = data.radon_number(r=models[0].get_num_of_states() + 2,
                                       h=1,
                                       d=data.train.shape[0])
        test_size = np.min([n_test, data.test.shape[0] - 1])
        #for m in models[0].px_model:
        #    preds = models[0].predict(m, test_size)
        #    accuracy = np.where(np.equal(preds[:, data.label_column], data.test_labels[:test_size]))[0].shape[0] / \
        #               data.test_labels[:test_size].shape[
        #                   0]
        #    print(str(accuracy))
    else:
        models = model
        dummy_model = SusyModel(data, path="SUSY")
        r = model[0].shape[0]
        d, r, h, n = data.radon_number(r=r + 2, h=h, d=data.train.shape[0])

    # Radon Machines
    methods = ['mean', 'radon', 'wa']
    aggregates = {k: [] for k in methods}
    for i, model in enumerate(models):
        logger.info("Aggregating Model No. " + str(i))
        aggr = [Mean(model),
                RadonMachine(model, r, h),
                WeightedAverage(model)]
        weights = None
        if isinstance(model, np.ndarray):
            weights = model
            model = dummy_model
        for name, aggregator in zip(methods, aggr):
            aggregates[name].append(
                aggregation_helper(model=model,
                                   aggregator=aggregator,
                                   data=data,
                                   weights=weights)
            )

    return models, aggregates, sampler


def susy_exp(name, Data, exp_loader, n, k, iters, h, epochs, n_models=None):
    save_path = os.path.join("experiments", name)
    model = None
    mask = None
    if exp_loader is not None:
        experiment_path = os.path.join(save_path, "_".join([str(n), str(iters), str(epochs)]) + "_" + str(exp_loader))
        seed = load_seed(experiment_path)
        model, mask = load_experiment(experiment_path)
    else:
        experiment_path, seed = save_seed(os.path.join(save_path, "_".join([str(n), str(iters), str(epochs)])))

    save_path = experiment_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    baseline(Data, name, iters, epochs, save_path, seed)
    data = Data(path=os.path.join("data", name), mask=mask, seed=seed)
    models, aggregate, sampler = run(data, save_path, model, n, k, iters, h, epochs, seed, n_models=n_models)

    if exp_loader is None:
        np.save(os.path.join(save_path, 'mask'), data.mask)
        np.save(os.path.join(save_path, 'splits'), sampler.split_idx)
        os.makedirs(os.path.join(save_path, "models"))
        for i, m in enumerate(models):
            np.save(os.path.join(save_path, str(i) + "_mask"), m.data.masks[i])
            for j, pxm in enumerate(m.px_model):
                pxm.save(os.path.join(save_path, "models", "k" + str(i) + "_n" + str(j) + "px"))
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
    name="COVERTYPE"
    data = CoverType
    result, agg = susy_exp(name=name,
                           Data=data,
                           exp_loader=None,
                           n=100,
                           k=10,
                           iters=50,
                           h=1,
                           epochs=1,
                           n_models=100)

    for key, aggregation_method in agg.items():
        print(key)
        for model in aggregation_method:
            print(str(model['acc']))
