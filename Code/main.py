from src.data.dataset import Dota2, Susy, Discretization
from src.model.aggregation import RadonMachine, Mean
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


def baseline(iters, epochs, dataset="SUSY", path=None, seed=None):
    models = None

    if os.path.isabs(dataset):
        data = Susy(path=dataset, seed=seed)
    else:
        data = Susy(path=os.path.join("data", dataset), seed=seed)

    sampler = Random(data, n_splits=1, seed=seed)
    if path is None or models is None:
        models = []
        for i, split in enumerate(sampler.split()):
            model = SusyModel(data, path=dataset)
            model.train(split=split, epochs=epochs, iters=iters)
            #predictions = model.predict()
            # marginals, A = model.px_model[0].infer()

            #accuracy = np.where(np.equal(predictions[i][:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
            #print("GLOBAL Model " + str(i) + ":" + str(accuracy))
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
    # models = np.load(os.path.join(path, "models.npy"))
    models = None
    mask = np.load(os.path.join(path, "mask.npy"))
    return models, mask


def load_seed(path):
    seed = np.load(os.path.join(path, "seed.npy"))
    return seed


def susy(exp_loader=None, n=100, k=10, iters=500, h=1, epochs=1):

    save_path = os.path.join("experiments", "susy")
    seed = None
    model = None
    if exp_loader is not None:
        experiment_path = os.path.join(save_path, "_".join([str(n), str(iters), str(epochs)]) + "_" + str(exp_loader))
        seed = load_seed(experiment_path)
        model, mask = load_experiment(experiment_path)
    else:
        experiment_path, seed = save_seed(os.path.join(save_path, "_".join([str(n), str(iters), str(epochs)])))

    save_path = experiment_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    baseline(iters, epochs, "SUSY", save_path, seed)

    models = []
    data = Susy(path="data/SUSY", seed=seed)
    if model is None:
        sampler = Random(data, n_splits=data.train.shape[0] / 1000, k=10, seed=seed)
        for split in sampler.split():
            model = SusyModel(data, path="SUSY")
            state_space = model.state_space + 1
            n_states = np.sum([state_space[i] * state_space[j] for i, j in model.edgelist])
            d, r, h, n = data.radon_number(r=n_states + 2, h=1, d=data.train.shape[0])
            model.train(split=split, epochs=1, iters=2, n_models=int(r))
            models.append(model)
    else:
        r = model.shape[0] + 2
        d, r, h, n = data.radon_number(r=r + 2, h=h, d=data.train.shape[0])

    # Radon Machines
    aggregates = []
    for i, model in enumerate(models):
        logger.info("Aggregating Model No. " + str(i))
        rm = RadonMachine(model, r, h)
        mean = Mean(model)
        aggregates.append(aggregation_helper(model, rm, data))
        aggregates.append(aggregation_helper(model, mean, data))

    if exp_loader is None:
        np.save(os.path.join(save_path, 'mask'), data.mask)
        np.save(os.path.join(save_path, 'splits'), split.split_idx)
    return models, aggregates


def aggregation_helper(model, aggregator, data):
    try:
        aggregator.aggregate(None)
        aggregate = aggregator.aggregate_models
        for aggregate_weights in aggregate:
            aggregate_model = px.Model(weights=aggregate_weights, graph=model.graph, states=model.state_space + 1)
            predictions = model.predict(aggregate_model)
            accuracy = np.where(np.equal(predictions[:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[
                0]
            logger.info(str(accuracy))
            return aggregate_weights, predictions, accuracy
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
    accuracy = np.where(np.equal(predictions[:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
    return model, radon_point, predictions, accuracy


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
    loader = 1582471048
    result, agg = susy()

    for aggregation_method in agg:
        weights = aggregation_method[0]
        predictions = aggregation_method[1]
        acc = aggregation_method[2]
