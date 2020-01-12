import os
import logging

import pandas as pd
import numpy as np
import networkx as nx

from src.conf.modes import ROOT_DIR, LOG_LEVEL
from src.data.util.chow_liu_tree import build_chow_liu_tree
from src.io.download import Download
from src.io.extract import Extract

# Logger Setup
logging.basicConfig(filename='data.log', level=LOG_LEVEL)


class Data:

    def __init__(self, url=None, path=None, name=None, data_dir=None):
        # Path Defs
        self.name = None
        self.url = None
        self.path = None

        if url is None and path is None and name is None:
            print(
                "Creating an Empty Data Set with no path or root information. Please specify either url, name or path to proceed.")

        # Data Defs
        self.data = None
        self.train = None
        self.test = None

        # Graph Defs
        self.chowliu = None

        if data_dir is None:
            print("No data directory provided defaulting to " + os.path.join(ROOT_DIR, 'data'))
            self.path = os.path.join(ROOT_DIR, 'data')
        else:
            self.path = data_dir
        if url:
            self.downloader = Download()
        elif path:
            self.extractor = Extract()
        elif name:
            self.name = name
            if not os.path.isdir(os.path.join(self.path, name)):
                os.makedirs(os.path.join(self.path, name))
            try:
                self.load()
            except FileNotFoundError as e:
                logging.debug("This is an Exception" + str(e))

        if self.train is None:
            # TODO:Generate Splits
            self.train = self.data
            self.train, self.test = self._train_test_split()

        if name is not None:
            self.root_dir = os.path.join(ROOT_DIR, "data", name, "model")
        else:
            self.root_dir = os.path.join(ROOT_DIR, "data", "NONAME", "model")
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def load(self):
        data_dir = os.path.join(self.path, self.name)
        os.chdir(data_dir)
        for file in os.listdir(data_dir):
            try:
                self.data[file] = pd.read_csv(file, header=None)
            except Exception as e:
                logging.debug("This is an Exception" + str(e))
        os.chdir(ROOT_DIR)

    def set_path(self, path):
        self.path = path

    def set_name(self, name):
        self.name = name

    def set_url(self, url):
        self.url = url

    def _train_test_split(self, ratio=0.8):
        n_data = self.data.shape[0]
        mask = np.random.rand(n_data) < ratio
        return self.data[mask], self.data[~mask]

    def vertices(self):
        if self.data is not None:
            vertices = self.data.columns
            return vertices

    def gen_chow_liu_tree(self):
        try:
            graph = nx.read_edgelist(os.path.join(self.root_dir, "chow_liu.graph"))
            self.chowliu = np.array([e for e in graph.edges], dtype=np.uint64)
            return
        except FileNotFoundError as fnf:
            print(str(fnf))
            print("Can not find edgelist in folder, generating new chow liu tree - this may take some time.")
        chow_liu_tree = build_chow_liu_tree(self.train.to_numpy(), len(self.vertices()))
        nx.write_edgelist(chow_liu_tree, os.path.join(self.root_dir, "chow_liu.graph"))
        nx.write_weighted_edgelist(chow_liu_tree, os.path.join(self.root_dir, "chow_liu_weighted.graph"))
        return chow_liu_tree.edges
