from __future__ import annotations
from enum import Enum
import numpy as np
from dataclasses import dataclass
from .. import *
from tools.log import LOGGER

V_MAX = 500
A_MAX = 4000
T_CLEAR = 2 * 120 / V_MAX


@dataclass()
class Graph:
    nodes: list[Node] = None
    edges: list[Edge] = None
    paths: dict[tuple[Node, Node], EdgePath] = None
    sections: dict[tuple[Node, Node], EdgeSection] = None
    item_sequences: tuple[ItemSequence] = None


class EdgePath:
    def __init__(self, edges: Edge | list[Edge] = None, items: set[Item] = None):
        if type(edges) is Edge:
            edges = [edges]
        self.edges: list[Edge] = edges
        self.items = items

    @property
    def start(self) -> Node:
        return self.edges[0].tail

    @property
    def goal(self) -> Node:
        return self.edges[-1].head

    @property
    def nodes(self) -> list[Node]:
        return [e.tail for e in self.edges] + [self.goal]

    def __repr__(self):
        if self.items:
            item_out = "["
            for i in self.items:
                item_out += "{0}, ".format(i)
            item_out = item_out[:-2] + "]"
        else:
            item_out = "[]"

        out = "EdgePath: {0:>3s}, ".format(item_out)
        if self.edges:
            for n in self.nodes:
                out += "{0} -> ".format(n)
        else:
            out += "[]    "
        return out[:-4]


class EdgeSection(EdgePath):
    def __init__(self, edges: list[Edge] = None, items: set[Item] = None, type: SectionType = None, flows_dict: dict[Item, float] = None):
        super().__init__(edges=edges, items=items)
        self.type = type
        self.flows_dict = flows_dict
        # TODO AllToAll could break this, as sections not in any same path might be added
        self.sections_prev: tuple[EdgeSection] = None
        self.sections_next: tuple[EdgeSection] = None

    def __repr__(self):
        if self.items:
            item_out = "["
            for i in self.items:
                item_out += "{0}, ".format(i)
            item_out = item_out[:-2] + "]"
        else:
            item_out = "[]"

        out = "Section_{0:11s}: {1:>6s}, ".format(self.type.name, item_out)
        if self.edges:
            if len(self.nodes) <= 2:
                out += "{0} -> {1}".format(self.start, self.goal)
            else:
                out += "{0} -> [{1:2d}] -> {2}".format(self.start, len(self.nodes) - 2, self.goal)
        else:
            out += "[]"
        return out


class Node:
    def __init__(self, pos: np.ndarray[float], id: int = None, can_wait: bool = True, can_skip: bool = False,
                 time_clear_fact: float = 1, edges: list[Edge] = None, edges_incoming: list[Edge] = None):
        self.pos = pos
        self.edges = edges or []
        self.edges_incoming = edges_incoming or []
        self.station: Station = None
        self.can_wait = can_wait
        self.can_skip = can_skip
        self.id = id
        self.time_clear = time_clear_fact * T_CLEAR

        self.section_sequences: dict[EdgeSection, tuple[ItemSequence]] = {}
        self.incoming_sequence_dependencies: list[ItemSequence] = []
        self.station_sequence: ItemSequence = None
        self.item_distribution: dict[Item, dict[EdgeSection, float]] = None

        self.mover_dependencies: set[Mover] = set()  # Contained movers wait for any ItemSequence to accept their item, if no valid sequence was found
        # Add if no contuniation sequence was found, reschedule after one contained sequence was commited
        self.mover_announcements: dict[Mover, ItemSequence] = {}

        self.mover_reserve_wait: Mover = None
        self.time_next_free = -np.inf

    @property
    def next_sections(self) -> tuple[EdgeSection]:
        return tuple(self.section_sequences.keys())

    def get_valid_sections(self, mover: Mover) -> list[EdgeSection]:
        forced_section = next((sec for sec, seq_tup in self.section_sequences.items() if any(seq.forced(mover=mover, node=self) for seq in seq_tup)), None)
        if forced_section:
            return [forced_section]

        allowed_sections = [sec for sec, seq_tup in self.section_sequences.items() if any(seq.allowed(mover=mover, node=self) for seq in seq_tup)]

        allowed_sections.sort(key=lambda sec: self.item_distribution[mover.item][sec])

        rejected_sections = [s for s in self.next_sections if s not in allowed_sections]
        out = "\t\tNext Section allowed {0}/{1}:".format(len(allowed_sections), len(self.next_sections))
        for section in allowed_sections:
            out += "\n\t\t\t{0}: {1}, flow {2:5.2f}, {3}".format(repr(section), [seq.next_item for seq in self.section_sequences[section]], self.item_distribution[mover.item]
                                                                 [section], "prio" if any(seq.prio() for seq in self.section_sequences[section]) else "")
        if rejected_sections:
            out += "\n\t\tRejected:"
        for section in rejected_sections:
            out += "\n\t\t\t{0}: {1}, flow {2:5.2f}, {3}".format(repr(section), [seq.next_item for seq in self.section_sequences[section]],
                                                                 self.item_distribution[mover.item].get(section, np.nan), "prio" if any(seq.prio() for seq in self.section_sequences[section]) else "")
        LOGGER.log_print_line("simulation", out)

        return allowed_sections

    def get_any_valid_section(self, mover: Mover) -> EdgeSection:
        return next(self.get_valid_sections(mover=mover))

    def announce_mover(self, mover: Mover, sequence: ItemSequence):
        self.mover_announcements[mover] = sequence

    def retract_announce_mover(self, mover: Mover, sequence: ItemSequence):
        if mover in self.mover_announcements and self.mover_announcements[mover] == sequence:
            self.mover_announcements.pop(mover)

    def commit(self, mover: Mover, section: EdgeSection, time: float) -> set[Mover]:
        forced_sequence = next((seq for seq in self.section_sequences[section] if seq.forced(mover=mover, node=self)), None)
        if forced_sequence:
            sequence = forced_sequence
        else:
            allowed_sequences = [seq for seq in self.section_sequences[section] if seq.allowed(mover=mover, node=self)]
            sequence = allowed_sequences[0]

        # LOGGER.log_print_line("simulation", "{0} comitting {1}".format(self, sequence))
        sequence.commit(mover=mover, node=self, time=time)
        section = next(section for section, sequences in self.section_sequences.items() if sequence in sequences)
        self.item_distribution[mover.item][section] += 1 / section.flows_dict[mover.item]

        if mover in self.mover_announcements:
            prev_sequence: ItemSequenceCritSplit | ItemSequenceCycle = self.mover_announcements[mover]
            if sequence != prev_sequence:
                del self.mover_announcements[mover]
                prev_sequence.restore_token(mover=mover)

        return sequence.get_blocked_dependencies()

    def commit_station(self, mover: Mover, time: float) -> set[Mover]:
        self.station_sequence.commit(mover=mover, node=self, time=time)

        if mover in self.mover_announcements:
            prev_sequence: ItemSequenceCritSplit | ItemSequenceCycle = self.mover_announcements[mover]
            del self.mover_announcements[mover]
            prev_sequence.restore_token(mover=mover)

        return self.station_sequence.get_blocked_dependencies()

    def reserve_pass(self, mover: Mover, action_time_start: float):
        assert self.mover_reserve_wait is None or self.mover_reserve_wait is mover
        self.mover_reserve_wait = None
        self.time_next_free = action_time_start + self.time_clear

    def reserve_wait(self, mover: Mover):
        assert self.mover_reserve_wait is None
        self.mover_reserve_wait = mover

    def add_dependency(self, blocked_mover: Mover):
        self.mover_dependencies.add(blocked_mover)

    def get_blocked_dependencies_copy(self) -> set[Mover]:
        mover_dependencies = self.mover_dependencies
        self.mover_dependencies = set()
        for prev_sequence in self.incoming_sequence_dependencies:
            mover_dependencies |= prev_sequence.get_blocked_dependencies()
        return mover_dependencies

    def __str__(self):
        return "Node_{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}]".format(self.pos[0], self.pos[1])

    def __repr__(self):
        return "Node {0} at [{1:4.0f},{2:4.0f}], {3} move edges, time_clear {4:4.2f}".\
            format("{0:2d}".format(self.id) if self.id is not None else 'N ',
                   self.pos[0], self.pos[1], len(self.edges), self.time_clear)


@dataclass
class Action:
    time_start: float
    edge: Edge
    acc: bool = True
    dec: bool = True
    mover: Mover = None
    _hash: int = None

    @property
    def node_start(self) -> Node:
        return self.edge.tail

    @property
    def node_goal(self) -> Node:
        return self.edge.head

    @property
    def edge_type(self) -> EdgeType:
        return self.edge.edge_type

    @property
    def edge_length(self) -> float:
        return self.edge.length

    @property
    def duration(self) -> float:
        return self.edge.length / V_MAX

    @property
    def time_goal(self) -> float:
        return self.time_start + self.duration

    def get_pos(self, time: float) -> np.ndarray[float]:
        if self.time_start > time or self.time_goal < time:
            raise Exception
        return self.edge.tail.pos + (self.edge.head.pos - self.edge.tail.pos) * (time - self.time_start) / self.duration

    def __lt__(self, other: Action):
        return self.time_start < other.time_start

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.node_goal, self.time_goal))
        return self._hash

    def __str__(self):
        return "({0}: {1:6.2f} - {2:6.2f}: {3})"\
            .format(self.mover, self.time_start, self.time_goal, self.edge)

    def __repr__(self):
        return "Action: Mover {0}: {1:6.2f} - {2:6.2f}: {3}"\
            .format(self.mover, self.time_start, self.time_goal, repr(self.edge))

    @staticmethod
    def list_to_str(actions: list[Action]) -> str:
        if not actions:
            return "No actions"
        out = "Mover {0} for {1:2d} actions: {2:6.2f}->{3:6.2f}: "\
            .format(actions[0].mover, len(actions), actions[0].time_start, actions[-1].time_goal)
        for action in actions[:-1]:
            out += "{0}@{1}__{2}__>".format(action.time_start, action.node_start, action.edge_type.name)
        out += "{0}@{1}__{2}__>{3}@{4}".format(actions[-1].time_start, actions[-1].node_start, actions[-1].edge_type.name, actions[-1].time_goal, actions[-1].node_goal)
        return out


class EdgeType(Enum):
    ANY = 0
    INTERACT = 1
    WAIT = 2
    MOVE = 3
    MOVE_NOWAIT = 4
    ENTER_STATION = 5
    LEAVE_STATION = 6


class Edge:
    def __init__(self, tail: Node, head: Node, edge_type: EdgeType = EdgeType.ANY, path_goals: list[Station] = None):
        self.tail = tail
        self.head = head
        self.edge_type = edge_type
        self.path_goals = path_goals

    @property
    def length(self) -> float:
        return np.linalg.norm(self.head.pos - self.tail.pos)

    def __str__(self):
        return "({0}: {1}->{2})".format(self.edge_type.name, self.tail, self.head)

    def __repr__(self):
        return "Edge {0}: {1}->{2}".format(self.edge_type.name, self.tail, self.head)
