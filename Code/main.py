from src.io.download import Download
from src.io.extract import Extract
from src.data.dota2 import Dota2
from src.model.dota2 import  Dota2 as Dota

from src.util.chow_liu_tree import build_chow_liu_tree
import pxpy as px
import numpy as np
import networkx as nx

def main():
    print("Hello World")
    downloader = Download()
    # downloader.start()
    extrator = Extract()

    # extrator.extract_all()

    dota_two = Dota2(name="DOTA2")
    dota_two.sample_match()
    dota_pgm = Dota(dota_two, path="DOTA2")
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
    px.train(data=dota_pgm.data.train.to_numpy(), in_model=p, iters=5)
    a = p.predict(observed=np.array([0, 1, 2], dtype=np.uint64))
    p.delete()
    print("We are done here this is for breakpoint stuff")


if __name__ == '__main__': main()
