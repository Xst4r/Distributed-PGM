from src.data.dataset import Dota2, Susy, Discretization
from src.model.aggregation import RadonMachine, Mean
from src.model.model import Dota2 as Dota
from src.model.model import Susy as SusyModel
from src.preprocessing.sampling import Random

from time import time
import datetime
import os
from shutil import copyfileobj

import numpy as np
import pxpy as px


def baseline(iters, epochs, dataset="SUSY", path=None, seed=None):
    models = None

    if os.path.isabs(dataset):
        data = Susy(path=dataset, seed=seed)
    else:
        data = Susy(path=os.path.join("data", dataset), seed=seed)

    sampler = Random(data, n_splits=1, seed=seed)
    model = SusyModel(data, path=dataset)

    if path is None or models is None:
        model.train(split=sampler, epochs=epochs, iters=iters)
        predictions = model.predict()
        marginals, A = model.px_model[0].infer()

        accuracy = np.where(np.equal(predictions[:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
        print("GLOBAL " + str(accuracy))
        os.makedirs(os.path.join(path, 'baseline'))
        model.px_model[0].save(os.path.join(path, 'baseline', 'px_model'))
        model.write_progress_hook(os.path.join(path, 'baseline'))
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


def experiment_from_file(path):
    models = np.load(os.path.join(path, path, "models"))
    mask = np.load(os.path.join(path, path, "mask"))
    return models, mask


def load_seed(path, experiment_identifier):
    seed = np.load(os.path.join(path, experiment_identifier, "seed"))
    return seed


def susy(load_experiment=None, n=100, iters=500, epochs=1, h=1):

    save_path = os.path.join("experiments", "susy")
    seed = None
    model = None
    if load_experiment is not None:
        seed = load_seed(load_experiment)
        model, mask = experiment_from_file(load_experiment)
        experiment_path = load_experiment
    else:
        experiment_path, seed = save_seed(os.path.join(save_path, "_".join([str(n), str(iters), str(epochs)])))

    save_path = experiment_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    baseline(iters, epochs, "SUSY", save_path, seed)
    data = Susy(path="data/SUSY", seed=seed)
    if model is None:
        model = SusyModel(data, path="SUSY")
        state_space = model.state_space + 1
        n_states = np.sum([state_space[i] * state_space[j] for i, j in model.edgelist])
        d, r, h, n = data.radon_number(r=n_states + 2, h=1, d=data.train.shape[0])
        split = Random(data, n_splits=data.train.shape[0] / 100)
        model.train(split=split, epochs=1, iters=100, n_models=r)
    else:
        r = model.shape[0] + 2
        d, r, h, n = data.radon_number(r=r + 2, h=h, d=data.train.shape[0])

    # Radon Machines
    radon_point = None
    predictions = None
    accuracy = None

    rm = RadonMachine(model, r, h)
    mean = Mean(model)
    aggregation_methods = [rm, mean]
    aggregates = []
    preds = []
    acc = []
    for aggregation_method in aggregation_methods:
        try:
            aggregation_method.aggregate(None)
            aggregate = aggregation_method.aggregate_models[0]
            aggregate_model = px.Model(weights=aggregate, graph=model.graph, states=model.state_space + 1)
            predictions = model.predict(aggregate_model)
            accuracy = np.where(np.equal(predictions[:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
            aggregates.append(aggregate)
            preds.append(predictions)
            acc.append(accuracy)
        except ValueError or TypeError as e:
            print(e)

    return model, aggregates, preds, acc


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
    loader = 1582371445
    # radon_cv("susy_n_100_iter_100_random_split.npy", "susy_n_100_iter_100_splits.npy")
    result, agg, labels, acc = susy()
    print(str(acc))
    if agg is not None:
        np.save("aggregate_model", agg)
    np.save("radon_predictions", labels[:,0])
    print(str(acc))
    n_models = len(result.px_model)
    idx = np.arange(n_models)
    np.random.shuffle(idx)
    test = np.ascontiguousarray(result.data.test)
    labels = result.data.test[0].to_numpy()
    test[:, 0] = -1
    local_acc = []
    for i in range(200):
        mod = result.px_model[idx[i]]
        loc_pred = mod.predict(test.astype(np.uint16))
        loc_acc = np.where(labels == loc_pred[:, 0])[0].shape[0] / labels.shape[0]
        local_acc.append((idx[i], loc_acc))
