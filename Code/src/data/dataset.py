import os
import logging

import pandas as pd
import numpy as np

from src.util.conf import ROOT_DIR, LOG_LEVEL
from src.io.download import Download
from src.io.extract import Extract

#Logger Setup
logging.basicConfig(filename='data.log',level=LOG_LEVEL)


class Data:

    def __init__(self, url=None, path=None, name=None, data_dir=None):
        self.name = None
        self.url = None
        self.path = None

        if url is None and path is None and name is None:
            print("Creating an Empty Data Set with no path or root information. Please specify either url, name or path to proceed.")
        self.data = {}
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
    def load(self):
        data_dir = os.path.join(self.path, self.name)
        os.chdir(data_dir)
        for file in os.listdir(data_dir):
            try:
                self.data[file] = pd.read_csv(file, header=None)
            except Exception as e:
                logging.debug("This is an Exception" + str(e))
        os.chdir(ROOT_DIR)
    def set_path(self):
        pass

    def set_name(self):
        pass

    def set_url(self):
        pass