from src.model.pgm import PGM

class Dota2(PGM):

    def __init__(self, data, weights=None, states=None, statespace=None):
        super(Dota2, self).__init__()