from src.data.dataset import Dota2, Susy, Discretization
from src.model.aggregation import RadonMachine, Mean
from src.model.model import Dota2 as Dota
from src.model.model import Susy as SusyModel
from src.preprocessing.split import Split

from time import time

import numpy as np
import pxpy as px


def susy():
    data = Susy(path="data/SUSY")

    full_model = SusyModel(data, path="SUSY")
    model = SusyModel(data, path="SUSY")



    full = Split(data, n_splits=1)
    full_model.train(split=full, epochs=2, iters=100)

    d, r, h, n = data.radon_number(r=full_model.global_weights[0].shape[0] + 2, h=1, d=data.train.shape[0])
    split = Split(data, n_splits=r ** h)


    model.train(split=split, epochs=2, iters=20)

    try:
        rm = RadonMachine(model, r, h)
        rm.aggregate(None)
        radon_point = rm.aggregate_models[0]
    except ValueError or TypeError as e:
        print("bla")

    try:
        mean = Mean(model)
        mean.aggregate(None)
        mean_theta = mean.aggregate_models[0]
    except ValueError or TypeError as e:
        print("bla")

    aggregate_model = px.Model(weights=radon_point, graph=model.graph, states=model.state_space)
    mean_model = px.Model(weights=mean_theta, graph=model.graph, states=model.state_space)

    predictions = model.predict(aggregate_model)
    avg_preds = model.predict(mean_model)
    accuracy = np.where(np.equal(predictions[:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
    avg_acc =  np.where(np.equal(avg_preds[:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
    print("AVG " + str(avg_acc))
    return model, radon_point, predictions, accuracy


def dota():
    data = Dota2(path="data/DOTA2")
    data.drop([23, 'ClusterID', 'GameMode', 'GameType'])
    model = Dota(data, path="DOTA2")

    d, r, h, n = data.radon_number(r=model.weights.shape[0] + 2)
    split = Split(data, n_splits=r ** h)
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
    for i in range(20):
        model = result.px_model[idx[i]]
        loc_pred = model.predict(test.astype(np.uint16))
        loc_acc = np.where(labels == loc_pred[:, 0])[0].shape[0] / labels.shape[0]
        local_acc.append((idx[i], loc_acc))
