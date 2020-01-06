import numpy as np

from src.model.pgm import PGM
from src.util.bijective_dict import BijectiveDict

class Dota2(PGM):

    def __init__(self, data, weights=None, states=None, statespace=None):

        self.data = data

        if states is None:
            self.states = self._states_from_data()
        if statespace is None:
            self.state_space = self._statespace_from_data()

        super(Dota2, self).__init__(data, weights, states, statespace)

        self.state_mapping = self._set_state_mapping()

    def _states_from_data(self):
        return len(self.data.train.columns)

    def _statespace_from_data(self):
        statespace = np.arange(self.states)
        for i, column in enumerate(self.data.train.columns):
            statespace[i] = np.unique(self.data.train[column]).shape[0]

        return statespace

    def _set_state_mapping(self):
        state_mapping = BijectiveDict()
        for i, column in enumerate(self.data.train.columns):
            state_mapping[str(column)] = i

        return state_mapping

    def edges_from_file(self, path):
        with open(path) as edgelist:
            n_edges = 0
            edges = edgelist.read()
            edges = edges.split(']')
            edge = np.empty(shape=(0, 2), dtype=np.uint64)
            for token in edges:
                token = token.strip("[").split()
                if len(token) < 2:
                    pass
                else:
                    clique = []
                    n_edges += (len(token) * (len(token) - 1))/2
                    for vertex in token:
                        clique.append(self.state_mapping[vertex])
                    for i, source in enumerate(clique):
                        for j in range(i, len(clique)):
                            if i != j:
                                edge = np.vstack((edge, np.array([source, clique[j]], dtype=np.uint64).reshape(1,2)))
            assert edge.shape[0] == n_edges
            self.add_edge(np.array(edge))