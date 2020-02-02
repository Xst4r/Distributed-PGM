from src.data.dataset import Dota2
from src.model.aggregation import RadonMachine
from src.model.model import Dota2 as Dota
from src.preprocessing.split import Split

from time import time

import numpy as np
import pxpy as px


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
    data = Dota2(path="data/DOTA2")

    test = data.test
    labels = np.copy(test['Result'].to_numpy())
    test['Result'] = -1
    test = np.ascontiguousarray(test.to_numpy().astype(np.uint16))
    model = Dota(data, path="DOTA2")

    # Prepare Radon Number and Splits
    d, r, h, n = data.radon_number(r=model.weights.shape[0] + 2)
    split = Split(data, n_splits=r ** h)
    print("Weights: " + str(model.weights.shape) + "\n" +
          "Radon Number " + str(r) + "\n")

    # Train Models
    model.train(split=split, iter=5)

    # Creating Coefficients for Linear Equations Ax = b
    # Each Theta (Parameter Vector) is a Column
    weights = []
    for px_model in model.px_model:
        weights.append(px_model.weights)
    weights = np.array(weights).T
    np.ascontiguousarray(weights.T)
    print(str(weights.shape))

    # Radon Machines
    try:
        rm = RadonMachine(model, r, h)
        rm.aggregate(None)
        radon_point = rm.aggregate_models[0]
    except ValueError or TypeError as e:
        print("bla")

    # Create new model with radon point and predict labels
    aggregate_model = px.Model(radon_point, model.graph, model.state_space)

    predictions = aggregate_model.predict(test)
    accuracy = np.where(np.equal(predictions[:,0], labels))[0].shape[0] / labels.shape[0]
    return model, radon_point, predictions, accuracy


if __name__ == '__main__':
    result, agg, labels, acc = main()
    np.save("aggregate_model", agg)
    np.save("radon_predictions", labels[:,0])
    print(str(acc))