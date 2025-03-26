from __future__ import annotations
import pickle
from dataclasses import dataclass

from .. import *
from tools.draw import SequenceDrawer


@dataclass
class Event:
    station: Station
    item: Item
    time_scheduled: float
    time_start: float
    duration: float
    is_loading: bool
    completes: bool
    mover: Mover = None

    @property
    def time_completes(self) -> float:
        return self.time_start + self.duration

    def __lt__(self, other: Event):
        return self.item < other.item if self.item != other.item else self.time_start < other.time_start

    def __str__(self):
        return "({0}: {1:2d}, at {2}, {3:6.2f} + {4:4.2f} -> {5:6.2f})"\
               .format("P" if self.station.is_provider else "R", self.item, self.station.pos, self.time_start, self.duration, self.time_completes)

    def __repr__(self):
        return str(self)


class Sequence:
    def __init__(self, sol: Solution, sce: Scenario):
        self.sol = sol
        self.sce = sce

        self.movers: list[Mover] = None
        self.events: dict[Station: list[Event]] = None

        self.drawer = SequenceDrawer(seq=self)

    def to_pickle(self, filename: str):
        self.drawer.clean_surfaces()
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, filename: str) -> Sequence:
        with open(filename, 'rb') as f:
            seq = pickle.load(f)
            return seq
