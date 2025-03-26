from __future__ import annotations
import numpy as np
from collections import deque
from tools.log import LOGGER
from itertools import cycle

from .. import *


class ItemSequence:
    def __init__(self, items: set[Item], node_entry: Node = None, node_exit: Node = None):
        self.items = items
        self.node_entry = node_entry
        self.node_exit = node_exit
        self.last_passed: float = -np.inf

    @property
    def next_item(self) -> Item:
        raise NotImplementedError

    @property
    def next_item_set(self) -> set[Item]:
        return set([self.next_item])

    @property
    def peek_sequence(self) -> tuple[Item]:
        raise NotImplementedError

    def forced(self, mover: Mover, node: Node) -> bool:
        return False

    def allowed(self, mover: Mover, node: Node) -> bool:
        return self.next_item == mover.item

    def prio(self) -> bool:
        return len(self.node_exit.mover_dependencies) > 0

    def commit(self, mover: Mover, node: Node, time: float):
        raise NotImplementedError

    def get_blocked_dependencies(self) -> set[Mover]:
        raise NotImplementedError

    def __repr__(self):
        return "ItemSequence Dummy"


class ItemSequenceSingle(ItemSequence):
    def __init__(self, item: Item, node_entry: Node, node_exit: Node):
        super().__init__(items=set([item]), node_entry=node_entry, node_exit=node_exit)
        self.item = item

    @property
    def next_item(self) -> Item:
        return self.item

    @property
    def peek_sequence(self) -> tuple[Item]:
        return tuple([self.item])

    def commit(self, mover: Mover, node: Node, time: float):
        self.last_passed = time

    def get_blocked_dependencies(self) -> set[Mover]:
        return self.node_entry.get_blocked_dependencies_copy()

    def __repr__(self):
        return "ItemSequenceSingle: {0}".format(self.item)


class ItemSequenceMultiple(ItemSequence):
    def __init__(self, sequence: tuple[Item], node_entry: Node, node_exit: Node):
        super().__init__(items=set(sequence), node_entry=node_entry, node_exit=node_exit)
        self.sequence: tuple[Item] = sequence

        self._idx_iter = iter(cycle(range(len(self.sequence))))
        self.next_idx = next(self._idx_iter)

    @property
    def next_item(self) -> Item:
        return self.sequence[self.next_idx]

    @property
    def peek_sequence(self) -> tuple[Item]:
        return self.sequence[self.next_idx:] + self.sequence[:self.next_idx]

    def commit(self, mover: Mover, node: Node, time: float):
        self.next_idx = next(self._idx_iter)
        self.last_passed = time

    def get_blocked_dependencies(self) -> set[Mover]:
        return self.node_entry.get_blocked_dependencies_copy()

    def __repr__(self):
        return "ItemSequenceMultiple: {0}, next {1}".format(self.sequence, self.peek_sequence)


class ItemSequenceCritSplit(ItemSequence):
    def __init__(self, next_section_sequences: tuple[ItemSequence], items: set[Item], node_entry: Node, node_exit: Node):
        super().__init__(items=items, node_entry=node_entry, node_exit=node_exit)
        self.next_section_sequences = next_section_sequences

        self.next_items_reserved = {i: 0 for i in self.items}

    @property
    def next_item(self) -> Item:
        return self.next_item_set

    @property
    def next_item_set(self) -> set[Item]:
        allowed = self.next_items_allowed()
        items = set(item for item, num_allowed in allowed.items() if num_allowed > self.next_items_reserved[item])
        return items

    @property
    def peek_sequence(self) -> tuple[tuple[Item]]:
        return tuple([s.peek_sequence for s in self.next_section_sequences])

    def next_items_allowed(self) -> dict[Item, int]:
        allowed = {i: 0 for i in self.items}
        for next_seq in self.next_section_sequences:
            for item in next_seq.next_item_set:
                allowed[item] += 1
        return allowed

    def allowed(self, mover: Mover, node: Node) -> bool:
        return mover.item in self.next_item_set
        # allowed = mover.item in self.next_item_set
        # if not allowed:
        #     LOGGER.log_print_line("simulation", "{0} blocked item {1}, allowed {2} , reserved {3}, seq {4}".format(self, mover.item, self.next_items_allowed(), self.next_items_reserved, self.peek_sequence))
        # return allowed

    def commit(self, mover: Mover, node: Node, time: float):
        assert self.allowed(mover, None)
        self.next_items_reserved[mover.item] += 1
        self.node_exit.announce_mover(mover=mover, sequence=self)
        self.last_passed = time
        # LOGGER.log_print_line("simulation", "{0} comitted item {1}, allowed {2}, reserved {3}, seq {4}".format(self, mover.item, self.next_items_allowed(), self.next_items_reserved, self.peek_sequence))

    def restore_token(self, mover: Mover):
        self.next_items_reserved[mover.item] -= 1
        # LOGGER.log_print_line("simulation", "{0} restored item {1}, allowed {2} , reserved {3}, seq {4}".format(self, item, self.next_items_allowed(), self.next_items_reserved, self.peek_sequence))

    def get_blocked_dependencies(self) -> set[Mover]:
        return self.node_entry.get_blocked_dependencies_copy()

    def __repr__(self):
        return "CritSplitSequence: {0}, next {1}".format([s.peek_sequence for s in self.next_section_sequences], self.next_item_set)


class ItemSequenceCritMerge(ItemSequence):
    def __init__(self, sequence: tuple[Item], node_exit: Node, nodes_entry: list[Node], node_deps: dict[Node, tuple[Node]]):
        super().__init__(items=set(sequence), node_exit=node_exit)
        self.sequence = sequence
        self.nodes_entry = nodes_entry  # (Node) of all nodes from which sequence can be entered - movers (blocked by this sequence) are subset of (deps of these nodes)
        self.node_deps = node_deps      # Node -> (Node) entry node -> nodes along path to exit node - reserve for mover when first entering
        self.mover_reservation: dict[Node, deque[Mover]] = None

        self._idx_iter = iter(cycle(range(len(self.sequence))))
        self.next_idx = next(self._idx_iter)

    @property
    def next_item(self) -> Item:
        return self.sequence[self.next_idx]

    def forced(self, mover: Mover, node: Node) -> bool:
        return len(self.mover_reservation[node]) > 0 and self.mover_reservation[node][0] == mover

    def allowed(self, mover: Mover, node: Node) -> bool:
        return self.next_item == mover.item and (len(self.mover_reservation[node]) == 0 or self.mover_reservation[node][0] == mover)

    def commit(self, mover: Mover, node: Node, time: float):
        # only process sequence once once when entering - otherwise just remove reservations
        if len(self.mover_reservation[node]) == 0:
            self.next_idx = next(self._idx_iter)
            self.last_passed = time

            out = "\tCritMerge {0} reserved: ".format(node)
            for next_node in self.node_deps[node]:
                self.mover_reservation[next_node].append(mover)
                out += "{0}: {1}, ".format(next_node, [m.id for m in self.mover_reservation[next_node]])
            LOGGER.log_print_line("simulation", out[:-2] if self.node_deps[node] else out + "None")

        else:
            assert self.mover_reservation[node][0] == mover
            self.mover_reservation[node].popleft()

    def get_blocked_dependencies(self) -> set[Mover]:
        mover_dependencies: set[Mover] = set()
        for node in self.nodes_entry:
            mover_dependencies |= node.get_blocked_dependencies_copy()
        return mover_dependencies

    def __repr__(self):
        return "CritMerge: {0}, next items {1}, res {2}".format(self.sequence, self.next_item, [(str(n), [m.id for m in q]) for n, q in self.mover_reservation.items() if len(q) > 0] if self.mover_reservation else None)


class ItemSequenceCycle(ItemSequence):
    def __init__(self, items: set[Item], next_section_sequences: tuple[ItemSequence], nodes_entry: tuple[Node], nodes_exit: dict[Item, tuple[Node]]):
        super().__init__(items=items, node_entry=None, node_exit=None)
        self.nodes_entry = nodes_entry
        self.nodes_exit = nodes_exit
        self.next_section_sequences = next_section_sequences

        self.next_items_reserved = {i: 0 for i in self.items}
        self.current_movers: set[Mover] = set()

    @property
    def next_item(self) -> Item:
        return self.next_item_set

    @property
    def next_item_set(self) -> set[Item]:
        allowed = self.next_items_allowed()
        items = set(item for item, num_allowed in allowed.items() if num_allowed > self.next_items_reserved[item])
        return items

    @property
    def peek_sequence(self) -> tuple[tuple[Item]]:
        return tuple([s.peek_sequence for s in self.next_section_sequences])

    def next_items_allowed(self) -> dict[Item, int]:
        allowed = {i: 0 for i in self.items}
        for next_seq in self.next_section_sequences:
            for item in next_seq.next_item_set:
                allowed[item] += 1
        return allowed

    def prio(self) -> bool:
        return False

    def allowed(self, mover: Mover, node: Node) -> bool:
        return mover in self.current_movers or mover.item in self.next_item_set

    def commit(self, mover: Mover, node: Node, time: float):
        if mover in self.current_movers:
            return

        self.next_items_reserved[mover.item] += 1
        self.current_movers.add(mover)
        for n in self.nodes_exit[mover.item]:
            n.announce_mover(mover=mover, sequence=self)
        self.last_passed = time

        LOGGER.log_print_line("simulation", "\tCycle {0} admitted mover {1} at {2}".format(self, mover, node))

    def restore_token(self, mover: Mover):
        self.next_items_reserved[mover.item] -= 1
        self.current_movers.remove(mover)
        for n in self.nodes_exit[mover.item]:
            n.retract_announce_mover(mover=mover, sequence=self)
        # LOGGER.log_print_line("simulation", "{0} restored item {1}, allowed {2} , reserved {3}, seq {4}".format(self, item, self.next_items_allowed(), self.next_items_reserved, self.peek_sequence))

    def get_blocked_dependencies(self) -> set[Mover]:
        mover_dependencies: set[Mover] = set()
        for node in self.nodes_entry:
            mover_dependencies |= node.get_blocked_dependencies_copy()
        return mover_dependencies

    def __repr__(self):
        return "Cycle: {0}, next items {1}".format([s.peek_sequence for s in self.next_section_sequences], self.next_item)
