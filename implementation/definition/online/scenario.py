from __future__ import annotations

from .. import *


class Scenario:
    def __init__(self, name: str, env: Environment, mover_num: int, time_max: float, seed: float = 0):
        self.name = name
        self.env = env
        self.mover_num = mover_num
        self.time_max = time_max
        self.seed = seed
