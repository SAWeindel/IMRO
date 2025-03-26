from __future__ import annotations
import numpy as np

from .. import *


class Item(int):
    def __str__(self) -> str:
        return super().__str__() if self != 0 else 'E'

    def __repr__(self) -> str:
        return super().__repr__() if self != 0 else 'E'

    @property
    def is_empty(self) -> bool:
        return self == 0

    @classmethod
    def empty(cls) -> Item:
        return Item(0)


class Station:
    def __init__(self, pos: np.ndarray[float], item_sequence: tuple[Item], item_type: str, period: float, duration: float, reset: float, is_provider: bool, size: np.ndarray[float] = None):
        self.pos = pos  # [mm,mm]
        self.node: Node = None
        self.item_sequence = item_sequence
        self.items = set(self.item_sequence)
        self.item_type = item_type  # ("any", "sequence")
        self.period = period
        self.duration = duration
        self.reset = reset
        self.size = size if size is not None else np.array([120, 120], dtype=float)
        self.is_provider = is_provider
        self.is_receiver = not is_provider

    @property
    def flow(self) -> float:
        return 1 / self.period

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __str__(self):
        return "({0}: [{1:4.0f},{2:4.0f}], items {3}, times [{4}, {5}, {6}])"\
               .format("P" if self.is_provider else "R", self.pos[0], self.pos[1], self.item_sequence, self.period, self.reset, self.duration)

    def __repr__(self):
        return "{0}: [{1:4.0f},{2:4.0f}] with items {3}, period {4}, reset {5}, duration {6}"\
               .format("Provider" if self.is_provider else "Receiver", self.pos[0], self.pos[1], self.item_sequence, self.period, self.reset, self.duration)
