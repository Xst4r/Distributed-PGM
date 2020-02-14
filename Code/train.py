import sys
import argparse


from typing import List

from src.data.dataset import Data
from src.model.model import Model
from src.preprocessing.sampling import Sampler


def run(args):
    parsed_args = parse_args(args)

    while True:
        next_model = None
        split = Sampler(next_model.train)
        train(next_model, split)


        write()

def train(model:Model, split:Sampler, agg, *kwargs):
    """

    Parameters
    ----------
    model :
    split :
    agg :
    kwargs :

    Returns
    -------

    Notes
    -----


    Examples
    --------
    """
    pass

def parse_args(args):
    pass


def write():
    write_model()
    write_plots()
    write_results()

def write_plots():
    pass

def write_model():
    pass

def write_results():
    pass

if __name__ == '__main__':
    run(sys.argv)