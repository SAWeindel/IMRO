from __future__ import annotations
import itertools
import numpy as np
from dataclasses import dataclass
from copy import copy
import json

from .. import *


@dataclass
class Network:
    primitives: list[Primitive] = None
    arcs: list[Arc] = None
    paths: dict[tuple[Primitive, Primitive], Path] = None
    sections: dict[tuple[Primitive, Primitive], Section] = None
    forces_comp: dict[Primitive: list] = None
    torques_comp: dict[Arc: list] = None
    pos_comp: dict[Primitive: np.ndarray[float]] = None
    next_id: int = None

    @property
    def n_movable(self):
        return sum(1 for p in self.primitives if p.fixed is False)

    def assign_ids(self, reassign: bool = False):
        if reassign:
            for i, p in enumerate(self.primitives):
                p.id = i
            self.next_id = len(self.primitives)
        else:
            if self.next_id is None:
                ids = tuple(p.id for p in self.primitives if p.id)
                self.next_id = max(ids) + 1 if ids else 0
            for prim in self.primitives:
                if not prim.id:
                    prim.id = self.next_id
                    self.next_id += 1

    def gen_primitives(self):
        primitives = set()
        for arc in self.arcs:
            primitives.add(arc.tail)
            primitives.add(arc.head)
        self.primitives = list(primitives)

    def gen_arcs(self):
        arcs = set()
        for prim in self.primitives:
            arcs.update(prim.arcs)
        self.arcs = list(arcs)

    def copy(self) -> Network:
        if self.arcs is None and self.primitives is None:
            return Network()
        if self.primitives is None:
            self.gen_primitives()
        elif self.arcs is None:
            self.gen_arcs()

        dup_primitives = [copy(prim) for prim in self.primitives]
        def dup_p(prim: Primitive): return dup_primitives[self.primitives.index(prim)]
        for iface, dup_iface in zip(self.primitives, dup_primitives):
            if type(iface) is not Iface:
                continue
            dup_iface.station = dup_p(iface.station)

        dup_arcs = [copy(arc) for arc in self.arcs]
        def dup_a(arc: Arc): return dup_arcs[self.arcs.index(arc)]
        for arc, dup_arc in zip(self.arcs, dup_arcs):
            dup_arc.tail = dup_p(arc.tail)
            dup_arc.head = dup_p(arc.head)

        for prim, dup_prim in zip(self.primitives, dup_primitives):
            if prim.incoming is not None:
                dup_prim.incoming = [dup_a(arc) for arc in prim.incoming]
            if prim.outgoing is not None:
                dup_prim.outgoing = [dup_a(arc) for arc in prim.outgoing]

        if self.paths is None:
            dup_paths = None
        else:
            dup_paths = {}
            for path in self.paths.values():
                dup_path_arcs = [dup_a(arc) for arc in path.arcs]
                start = dup_p(path.start)
                goal = dup_p(path.goal)
                items = path.items.copy()
                flows_dict = path.flows_dict.copy()
                dup_paths[(start, goal)] = Path(dup_path_arcs, items=items, flows_dict=flows_dict)

        for arc, dup_arc in zip(self.arcs, dup_arcs):
            dup_arc.paths = []
            for path in arc.paths:
                start = dup_p(path.start)
                goal = dup_p(path.goal)
                dup_arc.paths.append(dup_paths[(start, goal)])
            dup_arc.tail = dup_p(arc.tail)
            dup_arc.head = dup_p(arc.head)

        if self.sections is None:
            dup_sections = None
        else:
            dup_sections = {}
            section_assignment: dict[Section, Section] = {}
            for section in self.sections.values():
                dup_section_arcs = [dup_a(arc) for arc in section.arcs]
                start = dup_p(path.start)
                goal = dup_p(path.goal)
                items = section.items.copy()
                flows_dict = section.flows_dict.copy()
                dup_section = Section(dup_section_arcs, items=items, type=section.type, flows_dict=flows_dict)
                dup_sections[(start, goal)] = dup_section
                section_assignment[section] = dup_section

            for section, dup_section in section_assignment.items():
                dup_section.sections_prev = tuple(section_assignment[s] for s in section.sections_prev)
                dup_section.sections_next = tuple(section_assignment[s] for s in section.sections_next)

        forces_comp = {dup_p(prim): forces for prim, forces in self.forces_comp.items()} if self.forces_comp else None
        torques_comp = {dup_a(arc): torques for arc, torques in self.torques_comp.items()} if self.torques_comp else None

        return Network(primitives=dup_primitives, arcs=dup_arcs, paths=dup_paths, sections=dup_sections, forces_comp=forces_comp, torques_comp=torques_comp, next_id=self.next_id)

    def to_dict(self) -> dict:
        self.assign_ids()
        out = {
            'primitives': [p.to_dict() for p in self.primitives] if self.primitives else [],
            'arcs': [a.to_dict() for a in sorted(sorted(self.arcs, key=lambda a: a.head.id), key=lambda a: a.tail.id)] if self.arcs else [],
            'paths': [p.to_dict() for p in self.paths.values()] if self.paths else [],
            'sections': [s.to_dict() for s in self.sections.values()] if self.sections else []
        }
        return out

    @classmethod
    def from_dict(cls, x: dict, env: Environment) -> Network:
        prim_dict: dict[int, Primitive] = {}
        arc_dict: dict[tuple[int], Arc] = {}

        primitives: list[Primitive] = []
        for p_dict in x['primitives']:
            p_class = {'Prim': Primitive, 'IFace': Iface, 'Stion': PrimStation, 'Split': Split, 'Merge': Merge, 'Cross': Cross, 'A2A': Alltoall, 'Crner': Corner, 'Wait': Wait}[p_dict['type']]
            prim = p_class.from_dict(p_dict)
            primitives.append(prim)
            prim_dict[prim.id] = prim

        for p_dict in x['primitives']:
            prim = prim_dict[p_dict['id']]
            if type(prim) is Iface:
                prim.station = prim_dict[p_dict['station']]
            elif type(prim) is PrimStation:
                station = sorted(env.stations, key=lambda s: np.linalg.norm(s.pos - prim.pos))[0]
                prim.station = station

        arcs: list[Arc] = []
        for a_dict in x['arcs']:
            tail = prim_dict[a_dict['tail']]
            head = prim_dict[a_dict['head']]
            mover_size = np.array(a_dict['mover_size'], dtype=float)
            mover_rot_is_relative = a_dict['mover_rot_is_relative']
            mover_rot = a_dict['mover_rot']
            min_spaces = a_dict['min_spaces']

            arc = Arc(tail=tail, head=head, min_spaces=min_spaces, mover_rot=mover_rot, mover_rot_is_relative=mover_rot_is_relative, mover_size=mover_size)
            arc_dict[(tail.id, head.id)] = arc
            arcs.append(arc)

        paths: dict[tuple[Primitive, Primitive], Path] = {}
        for p_dict in x['paths']:
            p_arcs = [arc_dict[ids] for ids in zip(p_dict['primitives'][:-1], p_dict['primitives'][1:])]

            flows_dict = {env.get_item(i): f for i, f in p_dict['flows_dict'].items()}
            items = set(flows_dict.keys())

            path = Path(arcs=p_arcs, items=items, flows_dict=flows_dict)
            paths[(path.start, path.goal)] = path

            for arc in path.arcs:
                arc.paths.append(path)

        sections: dict[tuple[Primitive, Primitive], Path] = {}
        for s_dict in x['sections']:
            s_arcs = [arc_dict[ids] for ids in zip(s_dict['primitives'][:-1], s_dict['primitives'][1:])]
            flows_dict = {env.get_item(i): f for i, f in s_dict['flows_dict'].items()}
            items = set(flows_dict.keys())
            s_type = SectionType[s_dict['type']]

            section = Section(arcs=s_arcs, items=items, type=s_type, flows_dict=flows_dict)
            sections[(section.start, section.goal)] = section

        for s_dict in x['sections']:
            section = sections[(prim_dict[s_dict['primitives'][0]], prim_dict[s_dict['primitives'][-1]])]
            section.sections_prev = [sections[(prim_dict[start_id], section.start)] for start_id in s_dict['sections_prev']]
            section.sections_next = [sections[(section.goal, prim_dict[goal_id])] for goal_id in s_dict['sections_next']]

        return cls(primitives=primitives, arcs=arcs, paths=paths, sections=sections)

    @classmethod
    def from_json(cls, filename: str, env: Environment) -> Network:
        with open(filename, 'rb') as f:
            data = json.load(f)
        return Network.from_dict(x=data, env=env)

    def copy_reverse(self) -> Network:
        if self.arcs is None and self.primitives is None:
            return Network()
        if self.primitives is None:
            self.gen_primitives()
        elif self.arcs is None:
            self.gen_arcs()

        dup_primitives: list[Primitive] = []
        for prim in self.primitives:
            dup_prim = copy(prim)
            match type(prim):
                case Iface():
                    dup_prim.is_sink = not dup_prim.is_sink
                case Merge():
                    dup_prim = Split(pos=prim.pos.copy(), size=prim.size, rot=prim.rot, fixed=prim.fixed, id=prim.id, incoming=None, outgoing=None)
                case Split():
                    dup_prim = Merge(pos=prim.pos.copy(), size=prim.size, rot=prim.rot, fixed=prim.fixed, id=prim.id, incoming=None, outgoing=None)
            dup_primitives.append(dup_prim)

        def dup_p(prim: Primitive): return dup_primitives[self.primitives.index(prim)]
        for iface, dup_iface in zip(self.primitives, dup_primitives):
            if type(iface) is not Iface:
                continue
            dup_iface.station = dup_p(iface.station)

        dup_arcs = [copy(arc) for arc in self.arcs]
        def dup_a(arc: Arc): return dup_arcs[self.arcs.index(arc)]
        for arc, dup_arc in zip(self.arcs, dup_arcs):
            dup_arc.head = dup_p(arc.tail)
            dup_arc.tail = dup_p(arc.head)

        for prim, dup_prim in zip(self.primitives, dup_primitives):
            if prim.incoming is not None:
                dup_prim.outgoing = [dup_a(arc) for arc in prim.incoming]
            if prim.outgoing is not None:
                dup_prim.incoming = [dup_a(arc) for arc in prim.outgoing]

        if self.paths is None:
            dup_paths = None
        else:
            dup_paths = {}
            for path in self.paths.values():
                dup_path_arcs = list(reversed([dup_a(arc) for arc in path.arcs]))
                start = dup_p(path.goal)
                goal = dup_p(path.start)
                items = self.paths[(path.goal, path.start)].items.copy()
                flows_dict = self.paths[(path.goal, path.start)].flows_dict.copy()
                dup_paths[(start, goal)] = Path(dup_path_arcs, items=items, flows_dict=flows_dict)

        for arc, dup_arc in zip(self.arcs, dup_arcs):
            dup_arc.paths = []
            for path in arc.paths:
                start = dup_p(path.goal)
                goal = dup_p(path.start)
                dup_arc.paths.append(dup_paths[(start, goal)])
            dup_arc.tail = dup_p(arc.head)
            dup_arc.head = dup_p(arc.tail)

        return Network(primitives=dup_primitives, arcs=dup_arcs, paths=dup_paths, sections=None, forces_comp=None, torques_comp=None, next_id=self.next_id)

    def generate_corners_maxlength(self, size: np.ndarray[float], max_length: float):
        arcs = []
        for arc in self.arcs:
            num = np.floor(arc.length / max_length).astype(int)
            if num <= 0:
                continue

            pos_list = [arc.head.pos - (i + 1) * arc.vect / (num + 1) for i in reversed(range(num))]
            prims = [Corner(pos=pos, rot=0, size=size, fixed=False, incoming=None, outgoing=None) for pos in pos_list]

            new_arcs = [Arc(arc.tail, prims[0], mover_size=arc.mover_size)] +\
                       [Arc(tail, head, mover_size=arc.mover_size) for tail, head in zip(prims[:-1], prims[1:])]
            arc.update_tail(prims[-1])

            for path in arc.paths:
                path.insert_chain(new_arcs, next=arc)
            self.primitives.extend(prims)
            arcs.extend(new_arcs)
        self.arcs.extend(arcs)
        self.assign_ids()
        pass

    def generate_primitives_waiting(self, size: np.ndarray[float]):
        arcs = []
        for arc in self.arcs:
            num = np.floor(arc.length / arc.min_step_length - 1 + 0.001).astype(int)
            if num <= 0:
                continue

            pos_list = [arc.head.pos - (i + 1) * arc.dir * arc.min_step_length for i in reversed(range(num))]
            prims = [Wait(pos=pos, rot=0, size=size, fixed=True, incoming=None, outgoing=None) for pos in pos_list]

            new_arcs = [Arc(arc.tail, prims[0], mover_size=arc.mover_size)] +\
                       [Arc(tail, head, mover_size=arc.mover_size) for tail, head in zip(prims[:-1], prims[1:])]
            arc.update_tail(prims[-1])

            for path in arc.paths:
                path.insert_chain(new_arcs, next=arc)
            self.primitives.extend(prims)
            arcs.extend(new_arcs)
        self.arcs.extend(arcs)
        self.assign_ids()
        pass

    def generate_primitives_iface_arcs(self, size: np.ndarray[float], min_spaces: int = 0):
        self.arcs.extend([Arc(iface, iface.station, reverse_dir=iface.is_source, mover_size=size, min_spaces=min_spaces)
                          for iface in self.primitives if type(iface) is Iface])

    def generate_proto_paths(self, flows_dict: dict[tuple[Station, Station, Item], float]):
        self.paths = {}
        for arc in self.arcs:
            if type(arc.tail) is not Iface or type(arc.head) is not Iface:
                continue
            start = arc.tail.station
            start_arc = next(a for a in arc.tail.incoming if a.tail == start)
            goal = arc.head.station
            goal_arc = next(a for a in arc.head.outgoing if a.head == goal)

            if start.station.is_receiver:
                items = set([Item.empty()])
            else:
                items = start.station.items & goal.station.items

            # flows_dict: (start_station, goal_station, item) -> flow
            path_flows_dict = {i: f for (s, g, i), f in flows_dict.items() if s == start.station and g == goal.station}

            path = Path([start_arc, arc, goal_arc], items=items, flows_dict=path_flows_dict)
            self.paths[(start, goal)] = path

            start_arc.paths.append(path)
            arc.paths.append(path)
            goal_arc.paths.append(path)

    def generate_primitives_cross(self, size: np.ndarray[float], min_spaces: int = 0, use_merge_split: bool = False):
        def intersecting_pos(arc1: Arc, arc2: Arc) -> None | np.ndarray[float]:
            v1 = arc1.vect
            v2 = arc2.vect
            if np.cross(v1, v2) == 0:
                return None
            r = arc2.head.pos-arc1.tail.pos
            t1, t2 = np.linalg.solve(np.array([v1, v2]).T, r)
            if t1 < 0.01 or t1 > 0.99 or t2 < 0.01 or t2 > 0.99:
                return None
            return arc1.tail.pos + v1 * t1

        intersections: list[tuple[int, int, np.ndarray[float]]] = []
        intersecting_arcs: set[Arc] = set()
        for idx1, arc1 in enumerate(self.arcs):
            for arc2 in self.arcs[idx1 + 1:]:
                pos = intersecting_pos(arc1, arc2)
                if pos is not None:
                    intersections.append((arc1, arc2, pos))
                    intersecting_arcs.add(arc1)
                    intersecting_arcs.add(arc2)
        arc_segments = {arc: [arc] for arc in intersecting_arcs}

        prims = []
        for arc1, arc2, pos in intersections:
            if type(arc1.tail) is PrimStation or type(arc1.head) is PrimStation \
                    or type(arc2.tail) is PrimStation or type(arc2.head) is PrimStation:
                continue
            # has_solution = False
            for arc12, arc22 in itertools.product(arc_segments[arc1], arc_segments[arc2]):
                pos = intersecting_pos(arc12, arc22)
                if pos is None:
                    continue

                cross = Cross(pos=pos, rot=0, size=size, fixed=False)
                prims.append(cross)

                arc11, arc12 = arc12.insert_before_arc(cross, min_spaces=min_spaces)
                arc_segments[arc1] = [arc11] + arc_segments[arc1]
                self.arcs.append(arc11)

                arc21, arc22 = arc22.insert_before_arc(cross, min_spaces=min_spaces)
                arc_segments[arc2] = [arc21] + arc_segments[arc2]
                self.arcs.append(arc21)

                # has_solution = True
                break
            # assert has_solution

        self.primitives.extend(prims)
        self.assign_ids()

        if use_merge_split:
            for cross in prims:
                self.mod_split_cross(cross, min_spaces=min_spaces)

    def generate_primitives_split_merge(self, size: np.ndarray[float], sink_min_spaces: int = 0):
        ifaces = (p for p in self.primitives if type(p) is Iface)
        for iface in ifaces:
            arcs = []

            # Generate new Prims (Split or Merge) - Unbalanced tree structure with root iface. Placement along longest initial proto arc
            # Each prim connects prev prim (First: root iface) to one leaf iface and next prim. Last has two leaf ifaces.
            if iface.is_source:
                num_prims = len(iface.outgoing) - 1
                if num_prims == 0:
                    continue

                proto_arcs = sorted(iface.outgoing, key=lambda a: a.length)

                base_spacing = proto_arcs[-1].length / (num_prims + 1)
                if type(proto_arcs[-1].head) is Iface:
                    base_spacing *= len(iface.outgoing) / (len(iface.outgoing) + len(proto_arcs[-1].head.incoming) - 1)
                pos_list = [iface.pos + (i + 1) * proto_arcs[-1].dir * base_spacing for i in range(num_prims)]

                iface.outgoing = []
                prims = [Split(pos=pos, rot=0, size=size, fixed=False, incoming=None, outgoing=None) for pos in pos_list]
            else:
                num_prims = len(iface.incoming) - 1
                if num_prims == 0:
                    continue

                proto_arcs = sorted(iface.incoming, key=lambda a: a.length)

                base_spacing = proto_arcs[-1].length / (num_prims + 1)
                if type(proto_arcs[-1].tail) is Iface:
                    base_spacing *= len(iface.incoming) / (len(iface.incoming) + len(proto_arcs[-1].tail.outgoing) - 1)
                pos_list = [iface.pos - (i + 1) * proto_arcs[-1].dir * base_spacing for i in range(num_prims)]

                iface.incoming = []
                prims = [Merge(pos=pos, rot=0, size=size, fixed=False, incoming=None, outgoing=None) for pos in pos_list]

            # Connect first new prim with root iface
            arc = Arc(iface, prims[0], reverse_dir=iface.is_sink, min_spaces=sink_min_spaces*iface.is_sink, mover_size=size)
            arcs.append(arc)

            new_chains = {p_a: [arc] for p_a in proto_arcs}
            complete_chains: dict[Arc, list[Arc]] = {}

            for prim, next_prim, proto_arc in zip(prims[:-1], prims[1:], proto_arcs):
                # Connect prims with each other
                arc = Arc(prim, next_prim, reverse_dir=iface.is_sink, min_spaces=sink_min_spaces*iface.is_sink, mover_size=size)
                arcs.append(arc)

                # Connect prim with leaf iface
                if iface.is_source:
                    proto_arc.update_tail(prim)
                else:
                    proto_arc.update_head(prim)

                complete_chains[proto_arc] = new_chains.pop(proto_arc)
                for p_a in new_chains.keys():
                    if iface.is_source:
                        new_chains[p_a] = new_chains[p_a] + [arc]
                    else:
                        new_chains[p_a] = [arc] + new_chains[p_a]

            # Connect last prim with last two leaf iface
            if iface.is_source:
                proto_arcs[-1].update_tail(prims[-1])
                proto_arcs[-2].update_tail(prims[-1])
            else:
                proto_arcs[-1].update_head(prims[-1])
                proto_arcs[-2].update_head(prims[-1])

            complete_chains[proto_arcs[-1]] = new_chains.pop(proto_arcs[-1])
            complete_chains[proto_arcs[-2]] = new_chains.pop(proto_arcs[-2])

            for proto_arc, chain in complete_chains.items():
                for path in proto_arc.paths:
                    if iface.is_source:
                        path.insert_chain(new=chain, next=proto_arc)
                    else:
                        path.insert_chain(new=chain, prev=proto_arc)

            self.primitives.extend(prims)
            self.arcs.extend(arcs)

    def generate_sections(self):
        prims_split = (Split, Merge, Alltoall, PrimStation)

        sections: dict[tuple[Primitive, Primitive], Section] = {}
        sect_in: dict[Primitive, list[Section]] = {p: [] for p in self.primitives if type(p) in prims_split}
        sect_out: dict[Primitive, list[Section]] = {p: [] for p in self.primitives if type(p) in prims_split}

        for path in self.paths.values():
            arcs: list[Arc] = []
            for arc in path.arcs:
                arcs.append(arc)
                if type(arc.head) in prims_split:
                    prim_start = arcs[0].tail
                    prim_goal = arcs[-1].head
                    section_existing = next(filter(lambda s: s.goal == prim_goal, sect_out[prim_start]), None)
                    if section_existing:
                        section_existing.items |= path.items
                        for item, flow in path.flows_dict.items():
                            if item in section_existing.flows_dict:
                                section_existing.flows_dict[item] += flow
                            else:
                                section_existing.flows_dict[item] = flow
                    else:
                        items = path.items.copy()
                        flows_dict = path.flows_dict.copy()
                        section = Section(arcs=arcs, items=items, flows_dict=flows_dict)
                        sections[(section.start, section.goal)] = section
                        sect_out[prim_start].append(section)
                        sect_in[prim_goal].append(section)
                    arcs = []

        for section in sections.values():
            section.sections_prev = tuple(sect_in[section.start])
            section.sections_next = tuple(sect_out[section.goal])

        for section in sections.values():
            prim_start = section.start
            prim_goal = section.goal
            if type(prim_goal) is PrimStation:
                section.type = SectionType.IN_EXCL if len(prim_goal.incoming) == 1 else SectionType.IN_SHARED
            elif type(prim_start) is PrimStation:
                section.type = SectionType.OUT_EXCL if len(prim_start.outgoing) == 1 else SectionType.OUT_SHARED
            elif len(section.items) == 1:
                section.type = SectionType.ONE_TYPE
            else:
                section.type = SectionType.CRITICAL

        self.sections = sections

    def mod_add_corner(self, arc: Arc) -> Corner:
        corner = Corner(pos=(arc.head.pos + arc.tail.pos) / 2, size=arc.mover_size, rot=0, fixed=False)
        arc_in, arc = arc.insert_before_arc(primitive=corner)

        self.arcs.append(arc_in)
        self.primitives.append(corner)
        return corner

    def mod_add_double_corner(self, arc: Arc):
        corner2 = Corner(pos=(arc.head.pos + 2 * arc.tail.pos) / 3, size=arc.mover_size, rot=0, fixed=False)
        arc_mid, arc = arc.insert_before_arc(primitive=corner2)
        corner1 = Corner(pos=(2 * arc.head.pos + arc.tail.pos) / 3, size=arc.mover_size, rot=0, fixed=False)
        arc_in, arc_mid = arc_mid.insert_before_arc(primitive=corner1)

        self.arcs.append(arc_in)
        self.arcs.append(arc_mid)
        self.primitives.append(corner1)
        self.primitives.append(corner2)

    def mod_remove_corner(self, corner: Corner):
        arc_in = corner.incoming[0]
        arc_out = corner.outgoing[0]

        arc_out.absorb_prev()
        self.arcs.remove(arc_in)
        self.primitives.remove(corner)

    def mod_split_cross(self, cross: Cross, min_spaces: int = 0, arc_length: float = 0.25):
        arc1_in = cross.incoming[0]
        arc2_in = cross.incoming[1]
        arc1_out = cross.outgoing[0]
        arc2_out = cross.outgoing[1]

        vect = (arc1_in.dir + arc2_in.dir) / 2 * (cross.size[0] + cross.size[1]) / 2 * arc_length
        merge = Merge(pos=cross.pos - vect / 2, rot=cross.rot, size=cross.size, fixed=False)
        split = Split(pos=cross.pos + vect / 2, rot=cross.rot, size=cross.size, fixed=False)

        arc1_in.update_head(merge)
        arc2_in.update_head(merge)
        arc_mid = Arc(merge, split, mover_size=cross.size, min_spaces=min_spaces)
        self.arcs.append(arc_mid)
        arc1_out.update_tail(split)
        arc2_out.update_tail(split)

        for path in arc1_in.paths:
            path.insert_after_arc(prev=arc1_in, new=arc_mid)
        for path in arc2_in.paths:
            path.insert_after_arc(prev=arc2_in, new=arc_mid)

        self.primitives.remove(cross)
        self.primitives.append(merge)
        self.primitives.append(split)

    def post_process(self, grid_coarseness: float = 0.1):
        for prim in self.primitives:
            if prim.fixed:
                continue
            grid_size = prim.size * grid_coarseness
            prim_shift = prim.size * 0.5
            prim.pos = np.round((prim.pos + prim_shift) / grid_size) * grid_size - prim_shift
