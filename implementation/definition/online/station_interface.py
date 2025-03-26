from __future__ import annotations
import numpy as np
from collections import deque
from itertools import cycle
from random import Random

from .. import *


class EventGenerator:
    def __init__(self, station: Station, use_rand: bool = True, rand_seed: float = None, rand_min: float = 0.8, rand_max: float = 1.2, rand_dev: float = 0.1):
        self.station: Station = station

        self.use_rand = use_rand
        if self.use_rand:
            self.rand_seed = rand_seed
            _rand = Random(rand_seed)
            self.rand_gen = lambda: min(rand_max, max(rand_min, _rand.gauss(mu=1, sigma=rand_dev)))

        self.time_last_interaction: float = -np.inf
        self.item_iterator = iter(cycle(self.station.item_sequence))
        self.next_item = next(self.item_iterator)

    def __iter__(self):
        return self

    def next_event(self, time: float) -> Event:
        item = self.next_item
        self.next_item = next(self.item_iterator)

        time_scheduled = time
        time_start = max(time, self.time_last_interaction + self.station.reset)

        duration = self.station.duration
        if self.use_rand:
            duration *= self.rand_gen()
        completes = True

        self.time_last_interaction = time_start + duration

        return Event(station=self.station, item=item, time_scheduled=time_scheduled, time_start=time_start, duration=duration, is_loading=self.station.is_provider, completes=completes)


class StationInterface:
    def __init__(self, sce: Scenario, seq: Sequence, use_rand: bool = True):
        self.sce = sce
        self.seq = seq
        self.use_rand = use_rand

        self.stations = np.concatenate([self.sce.env.providers, self.sce.env.receivers])
        self.event_generators = {station: EventGenerator(station=station, use_rand=self.use_rand, rand_seed=self.sce.seed+i) for i, station in enumerate(self.stations)}
        self.open_events: dict[Station: deque[Event]] = {station: deque() for station in self.stations}
        self.seq.events = {station: [] for station in self.stations}

    def get_next_item(self, station: Station) -> Item:
        return self.event_generators[station].next_item

    def interact(self, station: Station, mover: Mover, time: float) -> Event:
        event = self.event_generators[station].next_event(time)
        event.mover = mover
        self.seq.events[station].append(event)
        return event
