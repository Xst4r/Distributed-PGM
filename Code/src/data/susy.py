#!/usr/bin/env python

"""
TODO: Description Here
"""
import os

import pandas as pd
import numpy as np

from .dataset import Data

from src.conf.settings import ROOT_DIR, get_logger

# Logger Setup
logger = get_logger()


class Susy(Data):

    def __init__(self, url=None, path=None, name=None, data_dir=None):


        super(Susy, self).__init__(url, path, name, data_dir)