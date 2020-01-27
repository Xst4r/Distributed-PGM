from src.io.download import Download
from src.io.extract import Extract
from src.data.dota2 import Dota2
from src.preprocessing.split import Split
from src.model.dota2 import  Dota2 as Dota
from src.data.susy import Susy

from src.data.susy import Discretization

from src.model.aggregation import radon_machine

import pxpy as px
import numpy as np

from time import  time

def main():
    #downloader = Download()
    #downloader.start()
    #extrator = Extract()

    #extrator.extract_all()

    data = Dota2(name="DOTA2")
    #susy = Susy(name="SUSY")
    #susy.discretize(Discretization.Quantile)
    model = Dota(data, path="DOTA2")
    data.sample_match()
    d, r, h, n = data.radon_number(r=model.weights.shape[0]+2)
    split = Split(data, devices=r**h)
    print("Weights: " + str(model.weights.shape) + "\n" +
          "Radon Number " + str(r) + "\n")
    model.train(split=split, iter=2)

    # Creating Weight Matrix for Ax = b
    weights = []

    start = time()
    lap = 0
    current = 0
    for px_model in model.px_model:
        if lap == 0 or current - lap > 20:
            lap = current
            if weights is not None:
                print(str(len(weights)))

        #weights = px_model.weights if weights is None else np.vstack((weights, px_model.weights.T))
        weights.append(px_model.weights)
        current = time()
    end = time()
    weights = np.array(weights).T
    print(str(end - start) + " s")
    np.ascontiguousarray(weights.T)
    print(str(weights.shape))


    # Radon Machines
    agg = radon_machine(weights, int(r), int(h))

    """
    test = Dota(dota_two, path="DOTA2")
    dota_pgm.edges_from_file("edges.graph")
    dota_pgm.gen_chow_liu_tree()
    # dota_two.to_csv('test.csv')

    # E = np.array([[0, 1],[0, 2], [0, 3], [0, 4]], dtype=np.uint64).reshape(4, 2)  # an edgelist with a single edge
    E = np.array([0, 1,
                  0, 2], dtype=np.uint64).reshape(2, 2)
    G = px.create_graph(dota_pgm.chow_liu_tree, nodes=None)

    # Sum of Size of all Clique State Spaces sum {c in C) |X_c|
    w = np.array([0.5, 0.6, 0.6, -0.3])  # edge weight vector (of an overcomplete model)
    p = px.Model(w, G, dota_pgm.state_space, stats=px.StatisticsType.overcomplete)

    data = dota_pgm.data.train.to_numpy()
    new_data = data - np.min(data, axis=0)[:]
    px.train(graph=G, data=new_data[0:50000], iters=1000)
    p.delete()
     """
    print("We are done here this is for breakpoint stuff")
    return model, agg

if __name__ == '__main__':
    model, agg = main()

