from __future__ import annotations
from collections import deque

from definition import *
from tools.log import LOGGER


class SequenceGenerator:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.section_sequences: dict[EdgeSection, list[ItemSequence]] = None

    def generate_sequences(self) -> tuple[ItemSequence]:
        self.section_sequences = {s: [] for s in self.graph.sections.values()}

        self.get_sequences_station()

        cycle_count = 0
        remaining_sections: deque[EdgeSection] = deque(filter(lambda s: s.goal.station is None, self.graph.sections.values()))
        while len(remaining_sections) > 0:
            section = remaining_sections.popleft()
            if any(len(self.section_sequences[s_n]) == 0 for s_n in section.sections_next):
                remaining_sections.append(section)
                cycle_count += 1
                if cycle_count >= len(remaining_sections):
                    cycles = self.detect_cycles(remaining_sections=remaining_sections)
                    for cyclic_sections in cycles:
                        self.new_cycle(sections=cyclic_sections)
                        for section in cyclic_sections:
                            remaining_sections.remove(section)
                continue
            cycle_count = 0
            self.get_sequences_section(section=section)

        for section, sequences in self.section_sequences.items():
            node_start = section.start
            node_start.section_sequences[section] = tuple(sequences)

            for crit_merge in (s for s in sequences if type(s) is ItemSequenceCritMerge):
                crit_merge.mover_reservation = {n: deque() for n in crit_merge.node_deps.keys()}

        for node in self.graph.nodes:
            items = set(item for sec in node.section_sequences.keys() for item in sec.items)
            node.item_distribution = {item: {sec: 0.0 for sec in node.section_sequences.keys() if item in sec.items} for item in items}

            if len(node.edges) > 1 and len(node.edges_incoming) == 1:  # split
                incoming_sequences = next(seq for sec, seq in self.section_sequences.items() if sec.goal == node)
                if len(incoming_sequences) > 1:
                    node.can_wait = False

        return self.section_sequences

    def detect_cycles(self, remaining_sections: deque[EdgeSection]):
        closed_sections: set[EdgeSection] = set(remaining_sections)
        open_sections: set[EdgeSection] = set()
        visited_sections: set[EdgeSection] = set()
        paths: list[list[Section]] = []
        cycles: list[list[Section]] = []

        while closed_sections:
            current_section = closed_sections.pop()
            paths.append([current_section])
            open_sections.add(current_section)

            while open_sections:
                current_section = open_sections.pop()
                visited_sections.add(current_section)

                next_sections = set(s for s in current_section.sections_next if s in remaining_sections)
                open_sections |= next_sections - visited_sections
                closed_sections -= next_sections

                old_paths = list(filter(lambda p: p[-1] == current_section, paths))
                for path in old_paths:
                    for next_section in next_sections:
                        if next_section in path:
                            cycle = path[path.index(next_section):]
                            cycles.append(cycle)
                            paths.remove(path)
                        else:
                            path.append(next_section)
                pass

        out = "Found {0} cycles:".format(len(cycles))
        for cycle in cycles:
            out += "\n\t{0}".format(cycle[0].start)
            for sec in cycle:
                out += " -> {0}".format(sec.goal)
        LOGGER.log_info(out)

        assert cycles
        return cycles

    def get_sequences_section(self, section: EdgeSection):
        node_entry = section.start
        node_exit = section.goal

        # Assignments below need all next sections to be completed
        if len(section.sections_next) > 1:  # Split
            next_sequences = [seq for sec_n in section.sections_next for seq in self.section_sequences[sec_n]]
            not_overlapping, overlap_sets = self.get_overlapping(next_sequences)

            for next_sequence in not_overlapping:
                match next_sequence:
                    case ItemSequenceSingle():
                        sequence = ItemSequenceSingle(item=self.extract(next_sequence.items), node_entry=node_entry, node_exit=node_exit)
                    case ItemSequenceMultiple():
                        sequence = ItemSequenceMultiple(sequence=next_sequence.sequence, node_entry=node_entry, node_exit=node_exit)
                    case ItemSequenceCritSplit():
                        sequence = ItemSequenceCritSplit(next_section_sequences=tuple([next_sequence]), items=next_sequence.items, node_entry=node_entry, node_exit=node_exit)
                        node_exit.incoming_sequence_dependencies.append(sequence)
                    case ItemSequenceCritMerge():
                        sequence = self.inherit_crit_merge(next_sequence=next_sequence, node_entry=node_entry, node_exit=node_exit)
                    case _:
                        raise Exception
                self.section_sequences[section].append(sequence)

            for overlap_set in overlap_sets:
                next_types = (type(seq) for seq in overlap_set)
                if all(t is ItemSequenceSingle for t in next_types):
                    sequence = ItemSequenceSingle(item=self.extract(self.extract(overlap_set).items), node_entry=node_entry, node_exit=node_exit)
                else:
                    items: set[Item] = set()
                    for next_sequence in overlap_set:
                        items |= next_sequence.items
                    sequence = ItemSequenceCritSplit(next_section_sequences=tuple(overlap_set), items=items, node_entry=node_entry, node_exit=node_exit)
                    node_exit.incoming_sequence_dependencies.append(sequence)
                self.section_sequences[section].append(sequence)

        else:  # Merge
            next_section = section.sections_next[0]

            parallel_sections = [sec for sec in next_section.sections_prev if sec != section]
            parallel_items: set[Item] = set()
            for p in parallel_sections:
                parallel_items |= p.items
            items_exclusive = section.items - parallel_items
            items_shared = section.items & parallel_items

            next_sequences = [seq for sec_n in section.sections_next for seq in self.section_sequences[sec_n]]
            next_sequences_exclusive = [seq for seq in next_sequences if len(seq.items & items_shared) == 0 and len(seq.items & items_exclusive) != 0]
            next_sequences_shared = [seq for seq in next_sequences if len(seq.items & items_shared) != 0]

            # LOGGER.log_info("PreMerge {0}\n\tparallel sections {1}\n\tnext {2}\n\tnext_exclusive {3}\n\tnext_shared {4}".format(section, parallel_sections, next_sequences, next_sequences_exclusive, next_sequences_shared))
            for next_sequence in next_sequences_exclusive:
                match next_sequence:
                    case ItemSequenceSingle():
                        sequence = ItemSequenceSingle(item=self.extract(next_sequence.items), node_entry=node_entry, node_exit=node_exit)
                    case ItemSequenceMultiple():
                        section_sequence = tuple(item for item in next_sequence.sequence if item in section.items)
                        if len(section_sequence) == 1:
                            sequence = ItemSequenceSingle(item=section_sequence[0], node_entry=node_entry, node_exit=node_exit)
                        else:
                            sequence = ItemSequenceMultiple(sequence=section_sequence, node_entry=node_entry, node_exit=node_exit)
                    case ItemSequenceCritMerge():
                        sequence = self.inherit_crit_merge(next_sequence=next_sequence, node_entry=node_entry, node_exit=node_exit)
                    case ItemSequenceCritSplit():
                        sequence = ItemSequenceCritSplit(next_section_sequences=tuple([next_sequence]), items=next_sequence.items, node_entry=node_entry, node_exit=node_exit)
                        node_exit.incoming_sequence_dependencies.append(sequence)
                    case ItemSequenceCycle():
                        section_sequence = tuple(item for item in next_sequence.items if item in section.items)
                        if len(section_sequence) == 1:
                            sequence = ItemSequenceSingle(item=section_sequence[0], node_entry=node_entry, node_exit=node_exit)
                        else:
                            sequence = ItemSequenceMultiple(sequence=section_sequence, node_entry=node_entry, node_exit=node_exit)
                    case _:
                        raise Exception
                self.section_sequences[section].append(sequence)

            for next_sequence in next_sequences_shared:
                match next_sequence:
                    case ItemSequenceSingle():
                        sequence = ItemSequenceSingle(item=self.extract(next_sequence.items), node_entry=node_entry, node_exit=node_exit)
                    case ItemSequenceMultiple():
                        new_next_sequence = self.new_crit_merge(sequence=next_sequence.sequence, node_entry=next_sequence.node_entry, node_exit=next_sequence.node_exit)
                        new_next_sequences = list(self.section_sequences[next_section])
                        new_next_sequences.remove(next_sequence)
                        new_next_sequences.append(new_next_sequence)
                        self.section_sequences[next_section] = tuple(new_next_sequences)
                        sequence = self.inherit_crit_merge(next_sequence=new_next_sequence, node_entry=node_entry, node_exit=node_exit)
                        LOGGER.log_info("Updated Section {0} sequence {1} to {2}".format(next_section, next_sequence, new_next_sequence))
                    case ItemSequenceCritMerge():
                        sequence = self.inherit_crit_merge(next_sequence=next_sequence, node_entry=node_entry, node_exit=node_exit)
                    case ItemSequenceCycle():
                        section_sequence = tuple(item for item in next_sequence.items if item in section.items)
                        if len(section_sequence) == 1:
                            sequence = ItemSequenceSingle(item=section_sequence[0], node_entry=node_entry, node_exit=node_exit)
                        else:
                            sequence = ItemSequenceMultiple(sequence=section_sequence, node_entry=node_entry, node_exit=node_exit)
                    case ItemSequenceCritSplit():
                        raise NotImplementedError
                    case _:
                        raise Exception
                self.section_sequences[section].append(sequence)

        LOGGER.log_info("Section {0} has sequence {1}".format(section, [s for s in self.section_sequences[section]]))

    def get_sequences_station(self):
        stations: dict[Node, list[EdgeSection]] = {n: [] for n in self.graph.nodes if n.station is not None}
        for section in self.graph.sections.values():
            node_exit = section.goal
            if node_exit in stations:
                stations[node_exit].append(section)

        for node_station, sections_incoming in stations.items():
            station = node_station.station
            if station.is_receiver:
                sequence_tuple = station.item_sequence
            else:
                sequence_tuple = tuple([Item.empty()])

            if len(sequence_tuple) == 1:
                station_sequence = ItemSequenceSingle(item=self.extract(sequence_tuple), node_entry=node_station, node_exit=node_station)
                pass
            else:
                station_sequence = ItemSequenceMultiple(sequence=sequence_tuple, node_entry=node_station, node_exit=node_station)

            node_station.station_sequence = station_sequence
            LOGGER.log_info("Station Node {0} has sequence {1}".format(node_station, station_sequence))

            if len(sections_incoming) == 1:
                section = sections_incoming[0]

                node_entry = section.start
                node_exit = section.goal

                if len(sequence_tuple) == 1:
                    sequence = ItemSequenceSingle(item=self.extract(sequence_tuple), node_entry=node_entry, node_exit=node_exit)
                else:
                    sequence = ItemSequenceMultiple(sequence=sequence_tuple, node_entry=node_entry, node_exit=node_exit)

                self.section_sequences[section] = [sequence]
                LOGGER.log_info("Section {0} has sequence {1}".format(section, sequence))
            else:
                for section in sections_incoming:
                    node_entry = section.start
                    node_exit = section.goal

                    parallel_sections = [sec for sec in sections_incoming if sec != section]
                    parallel_items: set[Item] = set()
                    for p in parallel_sections:
                        parallel_items |= p.items
                    items_exclusive = section.items - parallel_items
                    items_shared = section.items & parallel_items

                    if len(station_sequence.items & items_shared) == 0 and len(station_sequence.items & items_exclusive) != 0:  # exclusive
                        match station_sequence:
                            case ItemSequenceSingle():
                                sequence = ItemSequenceSingle(item=self.extract(station_sequence.items), node_entry=node_entry, node_exit=node_exit)
                            case ItemSequenceMultiple():
                                section_sequence = [item for item in station_sequence.sequence if item in section.items]
                                if len(section_sequence) == 1:
                                    sequence = ItemSequenceSingle(item=section_sequence[0], node_entry=node_entry, node_exit=node_exit)
                                else:
                                    sequence = ItemSequenceMultiple(sequence=section_sequence, node_entry=node_entry, node_exit=node_exit)
                    else:  # shared
                        match station_sequence:
                            case ItemSequenceSingle():
                                sequence = ItemSequenceSingle(item=self.extract(station_sequence.items), node_entry=node_entry, node_exit=node_exit)
                            case ItemSequenceMultiple():
                                new_station_sequence = self.new_crit_merge(sequence=station_sequence.sequence, node_entry=node_station, node_exit=node_station)
                                node_station.station_sequence = new_station_sequence
                                sequence = self.inherit_crit_merge(next_sequence=new_station_sequence, node_entry=node_entry, node_exit=node_exit)
                                LOGGER.log_info("Updated Station {0} sequence {1} to {2}".format(node_station, station_sequence, new_station_sequence))
                            case ItemSequenceCritMerge():
                                sequence = self.inherit_crit_merge(next_sequence=station_sequence, node_entry=node_entry, node_exit=node_exit)

                    self.section_sequences[section] = [sequence]
                    LOGGER.log_info("Section {0} has sequence {1}".format(section, sequence))

    def get_overlapping(self, sequences: list[ItemSequence]) -> tuple[set[ItemSequence], list[set[ItemSequence]]]:
        overlap_sets: list[set[ItemSequence]] = []
        overlapping: set[ItemSequence] = set()
        for idx1, seq1 in enumerate(sequences):
            for seq2 in sequences[idx1+1:]:
                if len(seq1.items & seq2.items) != 0:
                    overlapping.add(seq1)
                    overlapping.add(seq2)
                    for set_o in overlap_sets:
                        if seq1 in set_o or seq2 in set_o:
                            set_o.add(seq1)
                            set_o.add(seq2)
                            continue
                    overlap_sets.append(set((seq1, seq2)))
        not_overlapping = set(sequences) - overlapping
        return not_overlapping, overlap_sets

    def new_cycle(self, sections: list[EdgeSection]):
        items: set[Item] = set()
        for sec in sections:
            items |= sec.items

        next_section_sequences: list[Sequence] = []
        nodes_entry: set[Node] = set()
        nodes_exit: dict[Item, set[Node]] = {i: set() for i in items}

        for section in sections:
            sections_next = [s for s in section.sections_next if s not in sections]
            sections_prev = [s for s in section.sections_prev if s not in sections]

            if sections_next:
                shared_items = set()
                for section_next in sections_next:
                    next_section_sequences.extend(self.section_sequences[section_next])
                    shared_items |= section_next.items
                for item in shared_items:
                    nodes_exit[item].add(section.goal)

            if sections_prev:
                nodes_entry.add(section.start)

            for item in items:
                section.flows_dict[item] = 0.1
            section.items = items.copy()

        next_section_sequences = tuple(next_section_sequences)
        nodes_entry = tuple(nodes_entry)
        for item in nodes_exit:
            nodes_exit[item] = tuple(nodes_exit[item])

        sequence = ItemSequenceCycle(items=items, next_section_sequences=next_section_sequences, nodes_entry=nodes_entry, nodes_exit=nodes_exit)
        for node_set in nodes_exit.values():
            for node in node_set:
                node.incoming_sequence_dependencies.append(sequence)

        for section in sections:
            self.section_sequences[section].append(sequence)
            LOGGER.log_info("Section {0} has sequence {1}".format(section, sequence))
        pass

    def new_crit_merge(self, sequence: tuple[Item], node_entry: Node, node_exit: Node) -> ItemSequenceCritMerge:
        nodes_entry: list[Node] = [node_entry]
        node_deps: dict[Node, tuple[Node]] = {node_entry: tuple(), node_exit: tuple()}
        return ItemSequenceCritMerge(sequence=sequence, node_exit=node_exit, nodes_entry=nodes_entry, node_deps=node_deps)

    def inherit_crit_merge(self, next_sequence: ItemSequenceCritMerge, node_entry: Node, node_exit: Node) -> ItemSequenceCritMerge:
        next_sequence.nodes_entry.append(node_entry)
        nodes_dep_of_node_entry = tuple([node_exit]) + next_sequence.node_deps[node_exit]
        next_sequence.node_deps[node_entry] = nodes_dep_of_node_entry
        return next_sequence

    @staticmethod
    def extract(item_set: set[Item]) -> Item:
        for item in item_set:
            return item
