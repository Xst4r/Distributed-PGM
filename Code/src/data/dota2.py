import json
import os
import numpy as np
from os.path import join

from src.data.dataset import Data
from src.conf.settings import ROOT_DIR


class Dota2(Data):

    def __init__(self, url=None, path=None):
        self.heroes = {}
        self.hero_list = None
        super(Dota2, self).__init__(path, url)

        self.load_json()
        self.data_header()

    def load_json(self):
        with open(join(ROOT_DIR, 'data', 'DOTA2', 'heroes.json')) as file:
            data = json.load(file)

            heroes = list(data.values())
            self.heroes = {i['id']: i['name'] for i in heroes[0]}
            ordered_heroes = list(range(len(self.heroes) + 1))
            for i, (id, name) in enumerate(self.heroes.items()):
                ordered_heroes[id - 1] = name
            self.hero_list = ordered_heroes

    def data_header(self):
        header = ['Result', 'ClusterID', 'GameMode', 'GameType'] + self.hero_list
        self.data.reset_index()
        self.train.reset_index()
        self.test.reset_index()
        self.data.columns = header
        self.test.columns = header
        self.train.columns = header

    def sample_match(self):
        matches = self.data.shape[0]
        match_id = np.random.randint(0, high=matches - 1)

        match = self.data.iloc[match_id]

        result = match['Result']
        clusterid = match['ClusterID']
        gamemode = match['GameMode']
        gametype = match['GameType']

        picks = match.drop(labels=['Result', 'ClusterID', 'GameMode', 'GameType'])
        radiant = picks.where(lambda x: x == -1).dropna()
        dire = picks.where(lambda x: x == 1).dropna()

        if result == 1:
            winner = "Dire"
            print(winner + " won the match with \n" + str(list(dire.index)) + " against the radiant with \n" + str(
                list(radiant.index)))
        else:
            winner = "Radiant"
            print(winner + " won the match with \n" + str(list(radiant.index)) + " against the dire with \n" + str(
                list(dire.index)))

    def to_csv(self, path):
        self.train.to_csv(path)

    def __name__(self):
        return self.__class__.__name__

