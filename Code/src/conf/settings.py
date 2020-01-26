#!/usr/bin/env python

"""
Settings
--------
This file contains settings used in the project.
The settings are used to determine which methods are used for tasks
such as model-training, -creation and -aggregation.


Options
-------
ROOT_DIR: :String:  Determines the system path, where files, such as data and logs are created.
                    Has to be a valid system path.

LOG_LEVEL: Sets Logging level in the python logging facility valid options are:
           INFO, WARNING, ERROR, DEBUG

URLS: :class:`dict` Contains links in string format to data sets.
                    Links have to directly point towards a downloadable file.
                    When creating a downloader and starting the download, this object
                        determines the files downloaded.
                    {Name for the Data Set:Link to downloadable file}
CONF: :class:`dict` Contains a collection of Enums located in .modes.
                    Each element is an option, that sets the specified method for a task.

Functions
---------

set_config(agg, path, split)

set_log_level(loglevel)

set_url(url)

reset_config() : Resets config to default values

get_logger( LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            LOG_NAME       = '',
            LOG_FILE_INFO  = 'info.log',
            LOG_FILE_WARN  = 'warn.log',
            LOG_FILE_ERROR = 'err.log',
            LOG_FILE_DEBUG = 'dbg.log') :
            Returns a logger object used to log events of different
                severity in different files.
            It is advised to use the default options.
"""

# SUSY: https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz
# DOTA2: https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip
# covertype: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype


import os
import logging
from logging.handlers import RotatingFileHandler

from .modes.aggregation_type import  AggregationType
from .modes.graph_type import GraphType
from .modes.split_type import SplitType

# TODO: Create Config Class maybe ?

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DEBUG = True

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


def get_logger(
        LOG_FORMAT     = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        LOG_NAME       = '',
        LOG_FILE_INFO  = 'info.log',
        LOG_FILE_WARN  = 'warn.log',
        LOG_FILE_ERROR = 'err.log',
        LOG_FILE_DEBUG = 'dbg.log'):

    log = logging.getLogger(LOG_NAME)
    log_formatter = logging.Formatter(LOG_FORMAT)

    max_bytes = 2048
    backup_count = 6
    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)

    file_handler_info = RotatingFileHandler(LOG_FILE_INFO, mode='w', maxBytes=max_bytes, backupCount=backup_count)
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.INFO)
    log.addHandler(file_handler_info)

    file_handler_warning = RotatingFileHandler(LOG_FILE_WARN, mode='w', maxBytes=max_bytes, backupCount=backup_count)
    file_handler_warning.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.WARNING)
    log.addHandler(file_handler_warning)

    file_handler_error = RotatingFileHandler(LOG_FILE_ERROR, mode='w', maxBytes=max_bytes, backupCount=backup_count)
    file_handler_error.setFormatter(log_formatter)
    file_handler_error.setLevel(logging.ERROR)
    log.addHandler(file_handler_error)

    file_handler_debug = RotatingFileHandler(LOG_FILE_DEBUG, mode='w', maxBytes=max_bytes, backupCount=backup_count)

    log.setLevel(logging.INFO)

    return log


