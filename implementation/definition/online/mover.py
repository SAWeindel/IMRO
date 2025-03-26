from __future__ import annotations
import numpy as np
from enum import Enum
from tools.log import LOGGER

from .. import *


class MoverState(Enum):
    INTERACTING = 0
    BL_PATH = 2
    BL_ITEM = 3
    BL_NODE = 4
    BL_TIME = 5
    TO_GOAL = 6
    TO_BL_TIME = 7
    TO_BL_PATH = 8
    TO_BL_ITEM = 9
    TO_BL_NODE = 10
    TO_STATION = 11
    TO_INIT = 12


class BlockType(Enum):
    MOVER = 0
    NODE = 1
    TIME = 2
    ITEM = 3


class Mover:
    def __init__(self, id: int):
        self.id = id

        self.init_node: Node = None
        self.paths: list[EdgePath] = []
        self.items: list[tuple[float, Item]] = []
        self.actions: list[Action] = []
        self.states: list[tuple[float, MoverState]] = []

        self.last_station: Station = None
        self.next_time_to_plan = 0.0
        self.state: MoverState = None
        self.current_section: EdgeSection = None

        self.mover_dependencies: set[Mover] = set()

    @property
    def target_node(self) -> Node:
        return self.actions[-1].node_goal if self.actions else self.init_node

    @property
    def time_last_action(self) -> float:
        return self.actions[-1].time_goal if self.actions else 0

    @property
    def move_rem(self) -> tuple[Edge]:
        if self.current_section:
            idx = self.current_section.nodes.index(self.target_node)
            move_rem = tuple(self.current_section.edges[idx:])
            return move_rem
        elif self.paths:
            # init on path, follow path to init station
            path = self.paths[-1]
            idx = path.nodes.index(self.target_node)
            move_rem = tuple(path.edges[idx:])
            return move_rem
        else:
            return tuple()

    @property
    def item(self) -> Item:
        return self.items[-1][1] if self.items else None

    def set_state(self, state: MoverState, time: float = None):
        LOGGER.log_print_line("simulation", "\tState update {0} -> {1} @{2}".format(self.state.name, state.name, "{0:6.2f}".format(time) if time is not None else "None"))
        self.state = state
        if time is not None:
            self.states.append((time, state))

    def add_blocked_dependency(self, blocked_mover: Mover):
        self.mover_dependencies.add(blocked_mover)

    def peek_blocked_dependencies(self) -> set[Mover]:
        return self.mover_dependencies

    def get_blocked_dependencies_copy(self) -> set[Mover]:
        mover_dependencies = self.mover_dependencies
        self.mover_dependencies = set()
        return mover_dependencies

    def get_item(self, time: float) -> Item:
        if not self.items or self.items[0][0] > time:
            return Item.empty()
        if self.items[-1][0] <= time:
            return self.items[-1][1]
        return next((t[1] for t, tn in zip(self.items, self.items[1:]) if t[0] <= time and tn[0] > time))

    def get_state(self, time: float) -> MoverState | None:
        if not self.states or self.states[0][0] > time:
            return None
        if self.states[-1][0] <= time:
            return self.states[-1][1]
        return next((t[1] for t, tn in zip(self.states, self.states[1:]) if t[0] <= time and tn[0] > time))

    def get_action(self, time: float) -> Action | None:
        if not self.actions or self.actions[0].time_start > time:
            return None
        if self.actions[-1].time_start <= time:
            return self.actions[-1]
        return next((a for a, an in zip(self.actions, self.actions[1:]) if a.time_start <= time and an.time_start > time))

    def get_action_index(self, time: float) -> int | None:
        if not self.actions or self.actions[0].time_start > time:
            return None
        if self.actions[-1].time_start <= time:
            return len(self.actions) - 1
        return next((i for i in range(len(self.actions) - 1) if self.actions[i].time_start <= time and self.actions[i+1].time_start > time))

    def get_loc(self, time: float) -> Edge | Node | None:
        action = self.get_action(time)
        if action is None:
            return self.init_node

        if action.time_goal > time:
            return action.edge
        else:
            return action.node_goal

    def get_pos(self, time: float) -> np.ndarray[float] | None:
        action = self.get_action(time)
        if action is None:
            return self.init_node.pos

        if action.time_goal > time:
            return action.get_pos(time)
        else:
            return action.node_goal.pos

    def __lt__(self, other: Mover) -> bool:
        return self.next_time_to_plan < other.next_time_to_plan

    def __str__(self):
        return "{0:2d}".format(self.id)

    def __repr__(self):
        return "Mover {0:2d}, target {1}".format(self.id, self.target_node)
