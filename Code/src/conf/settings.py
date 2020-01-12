# SUSY: https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz
# DOTA2: https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip
# covertype: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype


import os
import logging
ROOT_DIR= os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

LOG_LEVEL = logging.DEBUG

URLS = {"DOTA2":"https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip",
        "SUSY":"https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz",
        "COVERTYPE": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"}