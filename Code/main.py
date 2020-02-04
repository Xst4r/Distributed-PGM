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
    #data.drop([23])
    model = Dota(data, path="DOTA2")

    # Prepare Radon Number and Splits
    d, r, h, n = data.radon_number(r=model.weights.shape[0] + 2)
    split = Split(data, n_splits=r**h)
    print("Weights: " + str(model.weights.shape) + "\n" +
          "Radon Number " + str(r) + "\n")

    # Train Models
    model.train(split=split, epochs=1, iters=2)

    #predictions = model.predict()
    #accuracy = np.where(np.equal(predictions[:, 0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
    #return model, None, predictions, accuracy
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
    accuracy = np.where(np.equal(predictions[:,0], data.test_labels))[0].shape[0] / data.test_labels.shape[0]
    return model, radon_point, predictions, accuracy


if __name__ == '__main__':
    result, agg, labels, acc = main()
    print(str(acc))
    if agg is not None:
        np.save("aggregate_model", agg)
    np.save("radon_predictions", labels[:,0])
    print(str(acc))