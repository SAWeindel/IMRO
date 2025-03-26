from __future__ import annotations
from enum import Enum
import numpy as np

from .. import *


class Path:
    def __init__(self, arcs: Arc | list[Arc] = None, items: set[Item] = None, flows_dict: dict[Item, float] = None):
        if type(arcs) is Arc:
            arcs = [arcs]
        self.arcs: list[Arc] = arcs
        self.items = items
        self.flows_dict = flows_dict

    @property
    def start(self) -> Primitive:
        return self.arcs[0].tail

    @property
    def goal(self) -> Primitive:
        return self.arcs[-1].head

    @property
    def primitives(self) -> list[Primitive]:
        return [a.tail for a in self.arcs] + [self.goal]

    @property
    def flow(self) -> float:
        return sum(self.flows_dict.values())

    def insert(self, arc: Arc, add_to_arc: bool = True):
        prims = self.primitives
        if arc.tail in prims:
            idx = prims.index(arc.tail)
        else:
            idx = prims.index(arc.head)
        self.arcs.insert(idx, arc)
        if add_to_arc:
            arc.paths.append(self)

    def insert_before_arc(self, new: Arc, next: Arc, add_to_arc: bool = True):
        idx = self.arcs.index(next)
        self.arcs.insert(idx, new)
        if add_to_arc:
            new.paths.append(self)

    def insert_after_arc(self, prev: Arc, new: Arc, add_to_arc: bool = True):
        idx = self.arcs.index(prev) + 1
        self.arcs.insert(idx, new)
        if add_to_arc:
            new.paths.append(self)

    def insert_chain(self, new: list[Arc], prev: Arc = None, next: Arc = None, add_to_arc: bool = True):
        idx = self.arcs.index(prev) + 1 if prev else self.arcs.index(next)
        self.arcs = self.arcs[:idx] + new + self.arcs[idx:]
        if add_to_arc:
            for arc in new:
                arc.paths.append(self)

    def remove_arc(self, arc: Arc):
        self.arcs.remove(arc)

    def to_dict(self) -> dict:
        out = {
            'primitives': tuple(p.id for p in self.primitives),
            'items': tuple(self.items),
            'flows_dict': self.flows_dict,
            '_nosplit': None
        }
        return out

    def __str__(self):
        if self.items:
            item_out = "["
            for i in self.items:
                item_out += "{0}, ".format(i)
            item_out = item_out[:-2] + "]"
        else:
            item_out = "[]"

        out = "Path: {0:>3s}, ".format(item_out)
        if self.primitives:
            if len(self.primitives) <= 2:
                out += "{0} -> {1}".format(self.start, self.goal)
            else:
                out += "{0} -> [{1:2d}] -> {2}".format(self.start, len(self.primitives) - 2, self.goal)
        else:
            out += "[]"
        return out

    def __repr__(self):
        if self.items:
            item_out = "["
            for i in self.items:
                item_out += "{0}, ".format(i)
            item_out = item_out[:-2] + "]"
        else:
            item_out = "[]"

        out = "Path: {0:>3s}@{1:4.2f}, ".format(item_out, self.flow)
        if self.primitives:
            for p in self.primitives:
                out += "{0} -> ".format(p)
        else:
            out += "[]    "
        return out[:-4]


class SectionType(Enum):
    OUT_EXCL = 0
    OUT_SHARED = 1
    IN_EXCL = 2
    IN_SHARED = 3
    ONE_TYPE = 4
    CRITICAL = 5


class Section(Path):
    def __init__(self, arcs: list[Arc] = None, items: set[Item] = None, type: SectionType = None, flows_dict: dict[Item, float] = None):
        super().__init__(arcs=arcs, items=items, flows_dict=flows_dict)
        self.type = type
        # TODO AllToAll could break this, as sections not in any same path might be added
        self.sections_prev: tuple[Section] = None
        self.sections_next: tuple[Section] = None

    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = self.type.name
        out['sections_prev'] = tuple(s.start.id for s in self.sections_prev)
        out['sections_next'] = tuple(s.goal.id for s in self.sections_next)
        return out

    def __repr__(self):
        if self.items:
            item_out = "["
            for i in self.items:
                item_out += "{0}, ".format(i)
            item_out = item_out[:-2] + "]"
        else:
            item_out = "[]"

        out = "Section_{0:11s}: {1:>6s}@{2:4.2f}, ".format(self.type.name, item_out, self.flow)
        if self.primitives:
            for p in self.primitives:
                out += "{0} -> ".format(p)
        else:
            out += "[]    "
        return out[:-4]


class Arc:
    def __init__(self, tail: Primitive, head: Primitive,
                 reverse_dir: bool = False, add_to_primitives: bool = True, min_spaces: int = 0,
                 mover_rot: float = 0, mover_rot_is_relative: bool = False, mover_size: np.ndarray[float] = None,
                 paths: list[Path] = None):
        self.tail = tail if not reverse_dir else head
        self.head = head if not reverse_dir else tail
        self.mover_size = mover_size if mover_size is not None else np.array([120, 120], dtype=float)

        self.mover_rot_is_relative = mover_rot_is_relative
        self._mover_rot = mover_rot
        self.min_spaces = min_spaces

        self.paths = paths or []

        if add_to_primitives:
            self.add_to_primitives()

    def add_to_primitives(self):
        self.tail.add_outgoing(self)
        self.head.add_incoming(self)

    def update_tail(self, new_tail: Primitive):
        if self.tail.outgoing and self in self.tail.outgoing:
            self.tail.outgoing.remove(self)
        new_tail.add_outgoing(self)
        self.tail = new_tail

    def update_head(self, new_head: Primitive):
        if self.head.incoming and self in self.head.incoming:
            self.head.incoming.remove(self)
        new_head.add_incoming(self)
        self.head = new_head

    def absorb_prev(self, update_paths: bool = True):
        assert len(self.tail.incoming) == 1
        old_tail = self.tail
        arc_prev = old_tail.incoming[0]
        self.update_tail(arc_prev.tail)
        arc_prev.update_tail(old_tail)  # cyclic - out of network
        if update_paths:
            for path in self.paths:
                path.remove_arc(arc_prev)

    def insert_before_arc(self, primitive: Primitive, min_spaces: int = 0, update_paths: bool = True) -> tuple[Arc, Arc]:
        paths = self.paths.copy()
        arc = Arc(tail=self.tail, head=primitive, mover_size=self.mover_size, min_spaces=min_spaces, paths=paths)
        self.update_tail(primitive)
        if update_paths:
            for path in self.paths:
                path.insert_before_arc(new=arc, next=self, add_to_arc=False)
        return arc, self

    def insert_after_arc(self, primitive: Primitive, min_spaces: int = 0, update_paths: bool = True) -> tuple[Arc, Arc]:
        paths = self.paths.copy()
        arc = Arc(tail=primitive, head=self.head, mover_size=self.mover_size, min_spaces=min_spaces, paths=paths)
        self.update_head(primitive)
        if update_paths:
            for path in self.paths:
                path.insert_after_arc(prev=self, new=arc, add_to_arc=False)
        return self, arc

    @property
    def fixed(self) -> bool:
        return self.tail.fixed and self.head.fixed

    @property
    def length(self) -> float:
        return np.linalg.norm(self.head.pos - self.tail.pos)

    @property
    def min_step_length(self) -> float:
        rot = np.deg2rad(self.rel_mover_rot % 180)
        b = np.arctan2(*self.mover_size)
        if b < rot and rot < np.pi - b:
            return self.mover_size[1] / abs(np.sin(rot))
        return self.mover_size[0] / abs(np.cos(rot))

    @property
    def min_length(self) -> float:
        return self.min_step_length * (self.min_spaces + 1)

    @property
    def vect(self) -> np.ndarray[float]:
        return self.head.pos - self.tail.pos

    @property
    def dir(self) -> np.ndarray[float]:
        return self.vect / self.length

    @property
    def norm(self) -> np.ndarray[float]:
        return np.array([-self.dir[1], self.dir[0]], dtype=float)

    @property
    def rot(self) -> float:
        return np.rad2deg(np.arctan2(*self.vect)) - 90

    @property
    def rot_rad(self) -> float:
        return np.arctan2(*self.vect) - np.pi / 2

    @property
    def rel_mover_rot(self) -> float:
        if self.mover_rot_is_relative:
            return self._mover_rot
        return self._mover_rot - self.rot

    @property
    def abs_mover_rot(self) -> float:
        if self.mover_rot_is_relative:
            return self._mover_rot + self.rot
        return self._mover_rot

    @property
    def width(self) -> float:
        rot = np.deg2rad(self.rel_mover_rot)
        return np.round(np.abs([np.cos(rot), np.sin(rot)]) @ self.mover_size, 5)

    # corners ordered [tail left,rear,right, head right, front, left]
    def get_corners(self, relative_rot: bool = True, relative_pos: bool = True):
        s0 = self.mover_size[0] / 2
        s1 = self.mover_size[1] / 2
        rot = np.deg2rad(self.rel_mover_rot)
        rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], dtype=float)
        corners = np.array([[s0, -s1], [-s0, -s1], [-s0, s1], [s0, s1]]) @ rot_matrix
        idx_top = np.argmin(corners[:, 1])
        idx_list = np.array([idx_top, idx_top + 1, idx_top + 2, idx_top + 2, idx_top + 3, idx_top], dtype=int) % 4
        corners_out = corners[idx_list]
        corners_out[3:] += [self.length, 0]
        if not relative_rot:
            rot = np.deg2rad(self.rot)
            rot_matrix = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], dtype=float)
            corners_out = corners_out @ rot_matrix
        if not relative_pos:
            corners_out += self.tail.pos
        return corners_out

    def to_dict(self) -> dict:
        out = {
            'tail': self.tail.id,
            'head': self.head.id,
            'mover_size': tuple((int)(x) for x in self.mover_size),
            'mover_rot_is_relative': self.mover_rot_is_relative,
            'mover_rot': self._mover_rot,
            'min_spaces': self.min_spaces,
            '_nosplit': None
        }
        return out

    def __copy__(self):
        return self.__class__(tail=None, head=None, add_to_primitives=False, min_spaces=self.min_spaces,
                              mover_rot=self._mover_rot, mover_rot_is_relative=self.mover_rot_is_relative,
                              mover_size=self.mover_size)

    def __str__(self):
        return "{0} -> {1}".format(self.tail, self.head)

    def __repr__(self):
        return "{0}->{1} R{2:4.0f}, L{3:3.0f}/{4:3.0f}, W{5:3.0f}, {6:1d} paths".format(repr(self.tail), repr(self.head), round(self.rot), self.length, self.min_length, self.width, len(self.paths))


class Primitive:
    def __init__(self, pos: np.ndarray[float], size: np.ndarray[float],
                 rot: float = None, fixed: bool = True, id: int = None,
                 incoming: list[Arc] = None, outgoing: list[Arc] = None):
        self.pos = pos
        self.size = size
        self.rot = rot
        self.fixed = fixed
        self.incoming = incoming if incoming is not None else []
        self.outgoing = outgoing if outgoing is not None else []
        self.id = id

    @property
    def arcs(self) -> list[Arc]:
        return self.incoming + self.outgoing

    @property
    def neighbors(self) -> list[Primitive]:
        return [a.tail for a in self.incoming] + [a.head for a in self.outgoing]

    @property
    def num_arcs(self) -> int:
        return len(self.incoming) + len(self.outgoing)

    @property
    def dir(self) -> np.ndarray[float]:
        r = np.deg2rad(self.rot) if self.rot else 0
        return np.round((np.sin(r), np.cos(r)), 5)

    def get_angles(self, ref_arc: Arc = None) -> list[tuple[Arc, float]]:
        ref_arc = ref_arc or self.incoming[0] or self.outgoing[0]
        ref_rot = ref_arc.rot if ref_arc.tail == self else (ref_arc.rot + 180) % 360
        arcs = [(arc, (arc.rot - ref_rot) % 360 if arc.tail == self else (arc.rot + 180 - ref_rot) % 360)
                for arc in self.arcs]
        return sorted(arcs, key=lambda a: a[1]) + [(ref_arc, 360)]

    def get_width_rot_rad(self, rot_rad: float) -> float:
        if self.rot is not None:
            rot_rad -= np.deg2rad(self.rot)
        return np.round(np.abs([np.cos(rot_rad), np.sin(rot_rad)]) @ self.size, 5)

    def get_corners(self, relative: bool = True) -> np.ndarray[float]:
        s0 = self.size[0] / 2
        s1 = self.size[1] / 2
        dir = self.dir
        rot_matrix = np.array([[dir[0], dir[1]], [-dir[1], dir[0]]], dtype=float)
        corners = np.array([[s0, -s1], [-s0, -s1], [-s0, s1], [s0, s1]]) @ rot_matrix
        if not relative:
            corners += self.pos
        return corners

    def add_incoming(self, arc: Arc):
        if self.incoming is None:
            self.incoming = [arc]
        else:
            self.incoming.append(arc)

    def add_outgoing(self, arc: Arc):
        if self.outgoing is None:
            self.outgoing = [arc]
        else:
            self.outgoing.append(arc)

    def to_dict(self) -> dict:
        out = {'type': "Prim",
               'id': self.id,
               'pos': tuple(self.pos.round(2)),
               'size': tuple((int)(x) for x in self.size),
               'rot': self.rot,
               'fixed': self.fixed,
               '_nosplit': None
               }
        return out

    @classmethod
    def from_dict(cls, x: dict) -> Primitive:
        return cls(id=x['id'], pos=np.array(x['pos']), size=np.array(x['size']), rot=x['rot'], fixed=x['fixed'])

    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id)

    def __str__(self):
        return " Prim_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return " Prim_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')


class Iface(Primitive):
    def __init__(self, pos: np.ndarray[float], size: np.ndarray[float],
                 is_source: bool, station: PrimStation = None,
                 rot: float = None, fixed: bool = True, id: int = None,
                 incoming: list[Arc] = None, outgoing: list[Arc] = None):
        super().__init__(pos=pos, size=size, rot=rot, fixed=fixed, id=id, incoming=incoming, outgoing=outgoing)
        self.station = station
        self.is_source = is_source

    @property
    def is_sink(self) -> bool:
        if self.is_source is None:
            return None
        return not self.is_source

    # def add_incoming(self, arc: Arc):
    #     if self.is_source == True: raise Exception("Iface is Source")
    #     super().add_incoming(arc=arc)

    # def add_outgoing(self, arc: Arc):
    #     if self.is_sink == True: raise Exception("Iface is Sink")
    #     super().add_outgoing(arc=arc)

    def get_angles(self, ref_arc: Arc = None) -> list[tuple[Arc, float]]:
        return super().get_angles(ref_arc=self.outgoing[0] if self.is_source else self.incoming[0])

    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = "IFace"
        out['station'] = self.station.id
        out['is_source'] = self.is_source
        return out

    @classmethod
    def from_dict(cls, x: dict) -> Iface:
        assert x['type'] == "IFace"
        return cls(id=x['id'], pos=np.array(x['pos']), size=np.array(x['size']), rot=x['rot'], fixed=x['fixed'], is_source=x['is_source'])

    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id, incoming=None, outgoing=None, station=None, is_source=self.is_source)

    def __str__(self):
        return "IFace_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return "IFace_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')


class PrimStation(Primitive):
    def __init__(self, pos: np.ndarray[float], size: np.ndarray[float],
                 station: Station = None,
                 rot: float = None, fixed: bool = True, id: int = None,
                 incoming: list[Arc] = None, outgoing: list[Arc] = None):
        super().__init__(pos=pos, size=size, rot=rot, fixed=fixed, id=id, incoming=incoming, outgoing=outgoing)
        self.station = station

    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = "Stion"
        return out

    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id, incoming=None, outgoing=None, station=self.station)

    def __str__(self):
        return "Stion_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return "Stion_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')


class Split(Primitive):
    def add_incoming(self, arc: Arc):
        self.incoming = [arc]

    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = "Split"
        return out

    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id, incoming=None, outgoing=None)

    def __str__(self):
        return "Split_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return "Split_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')


class Merge(Primitive):
    def add_outgoing(self, arc: Arc):
        self.outgoing = [arc]

    def get_angles(self, ref_arc: Arc = None) -> list[tuple[Arc, float]]:
        return super().get_angles(ref_arc=self.incoming[0])

    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = "Merge"
        return out

    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id, incoming=None, outgoing=None)

    def __str__(self):
        return "Merge_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return "Merge_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')


class Cross(Primitive):
    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id, incoming=None, outgoing=None)

    def __str__(self):
        return "Cross_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return "Cross_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')

    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = "Cross"
        return out


class Alltoall(Primitive):
    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = "A2A"
        return out

    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id, incoming=None, outgoing=None)

    def __str__(self):
        return "  A2A_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return "  A2A_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')


class Corner(Primitive):
    def add_incoming(self, arc: Arc):
        self.incoming = [arc]

    def add_outgoing(self, arc: Arc):
        self.outgoing = [arc]

    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = "Crner"
        return out

    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id, incoming=None, outgoing=None)

    def __str__(self):
        return "Crner_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return "Crner_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')


class Wait(Primitive):
    def add_incoming(self, arc: Arc):
        self.incoming = [arc]

    def add_outgoing(self, arc: Arc):
        self.outgoing = [arc]

    def to_dict(self) -> dict:
        out = super().to_dict()
        out['type'] = "Wait"
        return out

    def __copy__(self):
        return self.__class__(pos=self.pos.copy(), rot=self.rot, size=self.size, fixed=self.fixed, id=self.id, incoming=None, outgoing=None)

    def __str__(self):
        return " Wait_{0}".format("{0:2d}".format(self.id) if self.id is not None else "[{0:4.0f},{1:4.0f}|{2}]".format(self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N'))

    def __repr__(self):
        return " Wait_{0}@[{1:4.0f},{2:4.0f}|{3}]"\
            .format("{0:2d}".format(self.id) if self.id is not None else 'N ', self.pos[0], self.pos[1], '{0:3.0f}'.format(self.rot) if self.rot else '  N')


class ClusterType(Enum):
    PATH = 0
    SPLIT = 1
    MERGE = 2
    PARALLEL = 3
    CROSS = 4
    CROSS_BYPASS = 5
    CROSS_PARALLEL = 6


class Cluster:
    def __init__(self, cluster_type: ClusterType, edges: list[Edge]):
        self.cluster_type = cluster_type
        self.edges = edges

        self.primitives: list[Primitive] = None
        self.sources: list[Station] = None
        self.sinks: list[Station] = None
