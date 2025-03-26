from __future__ import annotations
import pickle

from .. import *
from tools.draw import SolutionDrawer


class Solution:
    def __init__(self, env: Environment, flow: Network = None, networks: list[Network] = None, graph: Graph = None, item_sequences: tuple[ItemSequence] = None):
        self.env = env

        self.flow = flow
        self.networks = networks or []
        self.graph = graph
        self.item_sequences = item_sequences

        self.reversible: bool = False

        self.drawer = SolutionDrawer(sol=self)

    def to_pickle(self, filename: str):
        self.drawer.clean_surfaces()
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filename: str) -> Solution:
        with open(filename, 'rb') as f:
            return pickle.load(f)
