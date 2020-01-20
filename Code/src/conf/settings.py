# SUSY: https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz
# DOTA2: https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip
# covertype: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype


import os
import logging

from .modes.aggregation_type import  AggregationType
from .modes.graph_type import GraphType
from .modes.split_type import SplitType

# TODO: Create Config Class maybe ?

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

LOG_LEVEL = logging.DEBUG

URLS = {"DOTA2":"https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip",
        "SUSY":"https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz",
        "COVERTYPE": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"}

CONF = {'AGGREGATION_TYPE': AggregationType.Mean,
        'GRAPH_TYPE': GraphType.ChowLiu,
        'SPLIT_TYPE': SplitType.Random
}


def set_config(agg, graph, split):
    CONF['AGGREGATION_TYPE'] = agg
    CONF['GRAPH_TYPE'] = graph
    CONF['SPLIT_TYPE'] = split


def set_log_level(level):
    LOG_LEVEL = level


def set_url(url):
    URLS = url

def reset_config():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    LOG_LEVEL = logging.DEBUG

    URLS = {"DOTA2": "https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip",
            "SUSY": "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz",
            "COVERTYPE": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"}

    CONF = {'AGGREGATION_TYPE': AggregationType.Mean,
            'GRAPH_TYPE': GraphType.ChowLiu,
            'SPLIT_TYPE': SplitType.Random
            }


