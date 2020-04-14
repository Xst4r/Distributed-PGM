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
import argparse
from pxpy import ModelType, SamplerType, GraphType
from src.data.metrics import squared_l2_regularization, prox_l1, default
from logging.handlers import RotatingFileHandler
from enum import IntEnum


class CovType(IntEnum):
    unif = 0
    random = 1
    fish = 2
    none = 3


class Config(object):

    def __init__(self):
        self.REGULARIZATION = default
        self.MODELTYPE = ModelType.integer
        self.SAMPLER = SamplerType.gibbs
        self.DEBUG = True
        self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.LOG_LEVEL = logging.DEBUG
        self.URLS = {"DOTA2":"https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip",
                     "SUSY":"https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz",
                     "COVERTYPE": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"}
        self.DATASET = None
        self.ARGS = None
        self.HOEFD_DELTA = None
        self.HOEFD_EPS = None
        self.GTOL = None
        self.TOL = None
        self.GRAPHTYPE = None
        self.CV = None
        self.EPOCHS = None
        self.COVTYPE = None

    def set_sampler(self, type):
        choices = {'gibbs': SamplerType.gibbs,
                   'map_perturb': SamplerType.apx_perturb_and_map}
        self.SAMPLER = choices[type]

    def set_regularization(self, type):
        choices = {'None': default,
                   'l1' : prox_l1,
                   'l2' : squared_l2_regularization}
        self.REGULARIZATION = choices[type]

    def set_model_type(self, type):
        choices = {'mrf': ModelType.mrf,
                   'integer': ModelType.integer}
        self.MODELTYPE = choices[type]

    def set_log_level(self, level):
        self.LOG_LEVEL = level

    def set_url(self, url):
        self.URLS = url

    def set_dataset(self, dataset):
        self.DATASET = dataset

    def set_cmd_args(self, cmd_args):
        self.ARGS = cmd_args

    def set_hoefding_eps(self, type):
        self.HOEFD_EPS = type

    def set_hoefding_delta(self, type):
        self.HOEFD_DELTA = type

    def set_gtol(self, type):
        self.GTOL = type

    def set_tol(self, type):
        self.TOL = type

    def set_graphtype(self, type):
        choices = {'chain': GraphType.chain,
                   'tree': GraphType.auto_tree,
                   'full': GraphType.full,
                   'star': GraphType.star}
        self.GRAPHTYPE = choices[type]

    def set_covtype(self, type):
        choices = {'unif': CovType.unif,
                   'random': CovType.random,
                   'fish': CovType.fish,
                   'none': CovType.none}
        self.COVTYPE = choices[type]

    def set_cv(self, type):
        self.CV = type

    def set_epochs(self, type):
        self.EPOCHS = type

    def get_logger(self,
            LOG_FORMAT     = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            LOG_NAME       = 'default_logger',
            LOG_FILE_INFO  = os.path.join('logs', 'info.log'),
            LOG_FILE_WARN  = os.path.join('logs', 'warn.log'),
            LOG_FILE_ERROR = os.path.join('logs', 'err.log'),
            LOG_FILE_DEBUG = os.path.join('logs', 'dbg.log')):

        log = logging.getLogger(LOG_NAME)

        # comment this to suppress console output
        if not os.path.isdir(os.path.join(".", "logs")):
            os.makedirs('logs')
        if not log.hasHandlers():
            log_formatter = logging.Formatter(LOG_FORMAT)
            loglevel = logging.DEBUG

            max_bytes = 2**24
            backup_count = 6
            log.setLevel(loglevel)
            handler = logging.StreamHandler()
            handler.setFormatter(log_formatter)
            log.addHandler(handler)
            log.setLevel(loglevel)
            log.handler_set = True

            file_handler_info = RotatingFileHandler(LOG_FILE_INFO, mode='w', maxBytes=max_bytes, backupCount=backup_count)
            file_handler_info.setFormatter(log_formatter)
            file_handler_info.setLevel(logging.INFO)
            log.addHandler(file_handler_info)

            file_handler_warning = RotatingFileHandler(LOG_FILE_WARN, mode='w', maxBytes=max_bytes, backupCount=backup_count)
            file_handler_warning.setFormatter(log_formatter)
            file_handler_warning.setLevel(logging.WARNING)
            log.addHandler(file_handler_warning)

            file_handler_error = RotatingFileHandler(LOG_FILE_ERROR, mode='w', maxBytes=max_bytes, backupCount=backup_count)
            file_handler_error.setFormatter(log_formatter)
            file_handler_error.setLevel(logging.ERROR)
            log.addHandler(file_handler_error)

            file_handler_debug = RotatingFileHandler(LOG_FILE_DEBUG, mode='w', maxBytes=max_bytes, backupCount=backup_count)
            file_handler_debug.setFormatter(log_formatter)
            file_handler_debug.setLevel(logging.DEBUG)
            log.addHandler(file_handler_debug)

        return log

    def setup(self, cmd_args):
        self.set_sampler(cmd_args.samp)
        self.set_regularization(cmd_args.reg)
        self.set_model_type(cmd_args.mt)
        self.set_cmd_args(cmd_args)
        self.set_hoefding_eps(cmd_args.hoefd_eps)
        self.set_hoefding_delta(cmd_args.hoefd_delta)
        self.set_gtol(cmd_args.gtol)
        self.set_tol(cmd_args.tol)
        self.set_graphtype(cmd_args.graphtype)
        self.set_cv(cmd_args.cv)
        self.set_covtype(cmd_args.covtype)

    def write_readme(self, path):
        with open(os.path.join(path, 'readme.md'), "w+") as readme:
            readme.write("DataSet : " + str(self.DATASET) + "\n")
            readme.write("ModelType : " + str(self.MODELTYPE) + "\n")
            readme.write("Sampler : " + str(self.SAMPLER) + "\n")
            readme.write("Regularization : " + str(self.REGULARIZATION.__name__) + "\n")
            if self.ARGS:
                for name, value in self.ARGS.__dict__.items():
                    readme.write(name + " : " + str(value)  + "\n")

CONFIG = Config()

def get_parser():

    parser = argparse.ArgumentParser(description="Distributed PGM Experiment Interface")

    parser.add_argument('--data',
                        metavar='DataSet',
                        type=str,
                        help='Name of a Dataset contained in /data',
                        default="COVERTYPE",
                        required=False)
    parser.add_argument('--maxiter',
                        metavar='Iterations',
                        type=int,
                        help='Maximum number of Iteration for each model.',
                        default=5000,
                        required=False)
    parser.add_argument('--load',
                        metavar='LoadExperiment',
                        type=int,
                        help='Identifier(Time in Seconds) found at the end of an experiment folder (e.g. 1583334301)',
                        required=False)
    parser.add_argument('--n_models',
                        metavar='NumModels',
                        type=int,
                        help='Number of Local (Distributed) Models',
                        default=10,
                        required=False)
    parser.add_argument('--cv',
                        metavar='CrossValidation',
                        type=int,
                        help='Number of Cross Validation Splits',
                        default=10,
                        required=False)
    parser.add_argument('--epoch',
                        metavar='Epochs',
                        type=int,
                        help="Number of Epochs (Rounds of Data retrieval on each local model) Increases the Amount of data "
                             "for each local model, each round.",
                        default=15,
                        required=False)
    parser.add_argument('--reg',
                        metavar='Regularization',
                        type=str,
                        help="Choose from available regularization options",
                        default='None',
                        choices=['None', 'l1', 'l2'],
                        required=False)
    parser.add_argument('--mt',
                        metavar='ModelType',
                        type=str,
                        help="Choose from available Modeltypes",
                        default='mrf',
                        choices=['mrf, integer'],
                        required=False)
    parser.add_argument('--samp',
                        metavar='Sampler',
                        type=str,
                        help="Choose from available Samplers",
                        default='gibbs',
                        choices=['gibbs, map_perturb'])

    parser.add_argument('--h',
                        type=int,
                        help="Number of Radon Machine Aggregation Steps. Each increment of h increases "
                             "the number of samples required exponentially by r**h",
                        default=1)

    parser.add_argument('--n_test',
                        metavar='TestSubset',
                        type=int,
                        help="Predicting a full test set, especially if it is large may take some time."
                             "Use this to reduce the number of predictions. "
                             "If n_test > test_size - test_size is chosen for prediction.",
                        default=10000)

    parser.add_argument('--hoefd_eps',
                        metavar='HoefdingDistance',
                        type=float,
                        help="Hyperparameter for the Hoefding Bound. Used to calculate the number of samples needed"
                             "to guarantee an upper bound on the distance with probability HoefdingProbability.",
                        default=1e-1)

    parser.add_argument('--hoefd_delta',
                        metavar='HoefdingProbability',
                        type=float,
                        help="Hyperparameter for the Hoefding Bound. Probability of suff. stats. having at most"
                             "distance of HoefdingDistance",
                        default=0.5)
    parser.add_argument('--gtol',
                        type=float,
                        help="Stopping criterion for the prox. gradient descent based on the gradient norm.",
                        default=1e-8)
    parser.add_argument('--tol',
                        type=float,
                        help="Stopping criterion for the prox. gradient descent based on objective rate of change.",
                        default=1e-8)

    parser.add_argument('--graphtype',
                        type=str,
                        help="GraphType for Probabilistic Graphical Models",
                        choices=["chain", "tree", "full", "star"],
                        default="tree")
    parser.add_argument('--covtype',
                        type=str,
                        help="Covariance Matrix Type for Model Parameter Sampling",
                        choices=["unif", "random", "fish", "none"],
                        default="fish")
    return parser
