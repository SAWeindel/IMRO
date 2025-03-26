from __future__ import annotations
from tools.log import LOGGER
from definition import *
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # noqa: E402
import pygame as pg


STATORSIZE = np.array([240, 240])

C_TEXT = "black"
C_FLOOR = "grey80"
C_ENV = "grey90"
C_LINES = "grey20"
C_FORBIDDEN = "grey20"
C_NODE = "black"
C_EDGE = "black"
C_MOVER = "grey50"
C_PROVIDER = pg.Color(0, 255, 0)
C_PROVIDER_BG = pg.Color(200, 255, 200)
C_RECEIVER = pg.Color(255, 0, 0)
C_RECEIVER_BG = pg.Color(255, 200, 200)
C_PRIM_BG = pg.Color(255, 255, 255, 200)
C_ITEMS = [C_MOVER, "cornflowerblue", "darkgoldenrod2", "darkorchid", "deeppink3", "antiquewhite3"]

FONTNAME = "consolas"
FONTSIZE = 12
pg.font.init()
FONT = pg.font.SysFont(name=FONTNAME, size=FONTSIZE)

FORCE_DICT = {"sum": "red",
              "primarc_arc": "green",
              "primarc_prim": "orange",
              "primprim": "purple",
              "arcs_repr": "yellow",
              "arcs_step": "cyan",
              "f_torque": "blue"}


class EnvironmentDrawer:
    def __init__(self, env: Environment):
        self.env = env
        self.env_surface_bg: pg.Surface = None

    def draw_environment(self, copy: bool = False) -> pg.Surface:
        if not self.env_surface_bg:
            self.env_surface_bg = pg.Surface(self.env.field_size, pg.SRCALPHA)
            self.env_surface_bg.fill(C_ENV)
            pg.draw.rect(self.env_surface_bg, C_LINES, (0, 0, *self.env.field_size), width=2)
            stator_dims = np.ceil(self.env.field_size / STATORSIZE).astype(int)
            for x, y in np.ndindex(tuple(stator_dims)):
                pg.draw.rect(self.env_surface_bg, C_LINES, (x * STATORSIZE[0], y * STATORSIZE[1], STATORSIZE[0],  STATORSIZE[1]), width=1)

            for provider in self.env.providers:
                r = pg.Rect(provider.pos - provider.size / 2, provider.size)
                pg.draw.rect(self.env_surface_bg, C_PROVIDER_BG, r, border_radius=10)
                pg.draw.rect(self.env_surface_bg, C_PROVIDER, r, width=5, border_radius=10)
            for receiver in self.env.receivers:
                r = pg.Rect(receiver.pos - receiver.size / 2, receiver.size)
                pg.draw.rect(self.env_surface_bg, C_RECEIVER_BG, r, border_radius=10)
                pg.draw.rect(self.env_surface_bg, C_RECEIVER, r, width=5, border_radius=10)

        if copy:
            return self.env_surface_bg.copy()
        else:
            return self.env_surface_bg

    def clean_surfaces(self):
        self.env_surface_bg = None


class SolutionDrawer:
    def __init__(self, sol: Solution):
        self.sol = sol
        self.env = self.sol.env

        self.node_size = int(np.min(self.env.grid_size) / 8)
        self.arrow_size = int(0.75 * self.node_size) * 2
        self.surface_arrow: pg.Surface = None

        self.sol_vid_writer: cv2.VideoWriter = None
        self.sol_vid_axes: plt.Axes = None

    def save_network_img(self, source: Graph | Network, text: str = "", filename: str = None,
                         draw_env: bool = True, draw_footprint: bool = True, draw_id: bool = True, draw_forces: bool = True):
        # if type(source) is Graph:
        if isinstance(source, sys.modules['definition'].Graph):
            surface = self.draw_graph(graph=source, draw_env=draw_env, draw_id=draw_id)
        # elif type(source) is Network:
        elif isinstance(source, sys.modules['definition'].Network):
            surface = self.draw_network(network=source, draw_env=draw_env, draw_footprint=draw_footprint, draw_id=draw_id, draw_forces=draw_forces)
        else:
            raise TypeError("source must be of type Graph or Network")

        if filename is None:
            if text is None or text == "":
                raise Exception("Name or Filename must be specified")
            filename = './output/' + text + '.png'

        if text is not None and text != "":
            s = FONT.render(text, True, "red")
            surface.blit(s, (10, self.env.field_size[1] - FONTSIZE - 1))
        pg.image.save(surface, filename)

    def draw_network(self, network: Network, draw_env: bool = True, draw_footprint: bool = True, draw_id: bool = True, draw_forces: bool = True) -> pg.Surface:
        if network is None:
            return
        surface = pg.Surface(self.env.field_size, pg.SRCALPHA)
        if draw_env:
            surface.blit(self.env.drawer.draw_environment(), (0, 0))

        surfs = []
        surfs_bg = []

        for arc in network.arcs:
            s, s_bg = self._draw_arc(arc, draw_footprint=draw_footprint)
            surfs.append(s)
            surfs_bg.append(s_bg)

        for prim in network.primitives:
            s, s_bg = self._draw_prim(prim, draw_footprint=draw_footprint, draw_id=draw_id)
            surfs.append(s)
            surfs_bg.append(s_bg)

        if draw_footprint:
            surface.blits(surfs_bg)
        surface.blits(surfs)
        if draw_forces and network.forces_comp:
            self._draw_forces(surface, network)

        return surface

    def draw_graph(self, graph: Graph, draw_env: bool = True, draw_id: bool = True) -> pg.Surface:
        if graph is None:
            return
        surface = pg.Surface(self.env.field_size, pg.SRCALPHA)
        if draw_env:
            surface.blit(self.env.drawer.draw_environment(), (0, 0))

        for edge in graph.edges:
            if edge.head == edge.tail:
                continue
            surface.blit(*self._draw_arrow(edge.tail.pos, edge.head.pos))

        for node in graph.nodes:
            self._draw_node(surface=surface, pos=node.pos, id=node.id if draw_id else None)

        return surface

    def clean_surfaces(self):
        self.surface_arrow = None
        self.env.drawer.clean_surfaces()

    def _draw_node(self, surface: pg.Surface, pos: np.ndarray[float], id: int = None):
        pg.draw.circle(surface, C_NODE, pos, self.node_size)
        if id is not None:
            s = FONT.render(str(id), True, "white")
            surface.blit(s, pos - (s.get_width() / 2, s.get_height() / 2))

    def _draw_arrow_head(self) -> pg.Surface:
        if not self.surface_arrow:
            self.surface_arrow = pg.Surface((self.arrow_size, self.arrow_size), pg.SRCALPHA)
            v = [(self.arrow_size - 1, self.arrow_size / 2),
                 (self.arrow_size - 1, self.arrow_size / 2 - 1),
                 (0, 0),
                 (self.arrow_size / 4, self.arrow_size / 2 - 1),
                 (self.arrow_size / 4, self.arrow_size / 2),
                 (0, self.arrow_size - 1)]
            pg.draw.polygon(self.surface_arrow, C_EDGE, v)
        return self.surface_arrow

    def _draw_arrow(self, tail_pos: np.ndarray[float], head_pos: np.ndarray[float], width: float = None) -> tuple[pg.Surface, pg.Rect] | tuple[tuple[pg.Surface, pg.Rect], tuple[pg.Surface, pg.Rect]]:
        edge_vect = head_pos - tail_pos
        edge_len = int(np.linalg.norm(edge_vect))
        edge_angle = np.rad2deg(np.arctan2(edge_vect[1], edge_vect[0]))
        box_width = width if (width and width > 0) else self.arrow_size

        s = pg.Surface((edge_len, box_width), pg.SRCALPHA)
        s_bg = pg.Surface((edge_len, box_width), pg.SRCALPHA)

        if width and width > 0:
            pg.draw.rect(surface=s_bg, color=C_PRIM_BG, rect=(0, 0, edge_len, box_width))
            pg.draw.rect(surface=s, color=C_EDGE, rect=(0, 0, edge_len, box_width), width=1)
        pg.draw.line(surface=s, color=C_EDGE, start_pos=(0, box_width / 2 - 2), end_pos=(edge_len, box_width / 2 - 2), width=4)
        s.blit(source=self._draw_arrow_head(), dest=(edge_len - 1.5 * self.node_size - self.arrow_size / 2, box_width / 2 - self.arrow_size / 2 - 1))

        s = pg.transform.rotate(s, -edge_angle)
        s_bg = pg.transform.rotate(s_bg, -edge_angle)
        r = s.get_rect(center=tail_pos + edge_vect / 2)

        return ((s, r), (s_bg, r)) if (width and width > 0) else (s, r)

    def _draw_arc(self, arc: Arc, draw_footprint: bool) -> tuple[tuple[pg.Surface, pg.Rect], tuple[pg.Surface, pg.Rect]]:
        edge_vect = arc.head.pos - arc.tail.pos
        edge_angle = arc.rot
        edge_len = int(np.linalg.norm(edge_vect))

        corners = np.round(arc.get_corners(relative_rot=True, relative_pos=True))
        corner_vect = np.linalg.norm(corners[:2], axis=1)
        corners += corner_vect

        box_len = edge_len + 2 * corner_vect[0]
        box_width = 2 * corner_vect[1]
        s = pg.Surface((box_len, box_width), pg.SRCALPHA)
        s_bg = pg.Surface((box_len, box_width), pg.SRCALPHA)

        pg.draw.line(surface=s, color=C_EDGE, start_pos=corner_vect + (0, -2), end_pos=corner_vect + (edge_len, - 2), width=4)
        s.blit(source=self._draw_arrow_head(), dest=corner_vect + (edge_len - 1.5 * self.node_size - self.arrow_size / 2, - self.arrow_size / 2 - 1))
        s = pg.transform.rotate(s, edge_angle)

        if draw_footprint:
            s_lines = pg.Surface((box_len, box_width), pg.SRCALPHA)
            pg.draw.line(surface=s_lines, color=C_EDGE, start_pos=corners[0], end_pos=corners[5], width=1)
            pg.draw.line(surface=s_lines, color=C_EDGE, start_pos=corners[2] - (0, 1), end_pos=corners[3] - (0, 1), width=1)
            s.blit(pg.transform.rotate(s_lines, edge_angle), (0, 0))
            pg.draw.polygon(surface=s_bg, color=C_PRIM_BG, points=corners)
            s_bg = pg.transform.rotate(s_bg, edge_angle)

        r = s.get_rect(center=arc.tail.pos + edge_vect / 2)
        return ((s, r), (s_bg, r))

    def _draw_prim(self, prim: Primitive, draw_footprint: bool, draw_id: bool) -> tuple[tuple[pg.Surface, pg.Rect], tuple[pg.Surface, pg.Rect]]:
        s = pg.Surface(prim.size, pg.SRCALPHA)
        s_bg = pg.Surface(prim.size, pg.SRCALPHA)

        if draw_footprint:
            pg.draw.rect(s_bg, C_PRIM_BG, ((0, 0), prim.size), border_radius=10)
            # if type(prim) is PrimStation:
            if isinstance(prim, sys.modules['definition'].PrimStation):
                if prim.station.is_provider:
                    pg.draw.rect(s_bg, C_PROVIDER_BG, ((0, 0), prim.size), border_radius=10)
                    pg.draw.rect(s_bg, C_PROVIDER, ((0, 0), prim.size), width=5, border_radius=10)
                else:
                    pg.draw.rect(s_bg, C_RECEIVER_BG, ((0, 0), prim.size), border_radius=10)
                    pg.draw.rect(s_bg, C_RECEIVER, ((0, 0), prim.size), width=5, border_radius=10)
            else:
                pg.draw.rect(s_bg, C_NODE, ((0, 0), prim.size), width=1, border_radius=10)

        self._draw_node(surface=s, pos=prim.size/2, id=prim.id if draw_id else None)

        r = s.get_rect(center=prim.pos)
        return ((s, r), (s_bg, r))

    def _draw_forces(self, surface: pg.Surface, network: Network, fact: float = 5):
        for prim in network.primitives:
            if prim not in network.forces_comp:
                continue

            force_sum = np.zeros(2, dtype=float)
            for name, force in network.forces_comp[prim]:
                force_sum += force
                color = FORCE_DICT.get(name.split(':')[0], "grey")
                pg.draw.line(surface, color, start_pos=prim.pos, end_pos=prim.pos + force * fact, width=4)
                pg.draw.circle(surface, color, prim.pos + force * fact + (1, 1), 4)
            if np.linalg.norm(force_sum) * fact > 5:
                pg.draw.line(surface, "red", start_pos=prim.pos, end_pos=prim.pos + force_sum * fact - force_sum / np.linalg.norm(force_sum) * 4, width=2)
            pg.draw.circle(surface, "red", prim.pos + force_sum * fact + (1, 1), radius=6, width=2)


class OptimizerDrawer:
    def __init__(self, sol: Solution, filename: str = './output/force_placement.avi'):
        self.sol = sol
        self.env = self.sol.env

        self.sol_vid_writer: cv2.VideoWriter = None
        self.sol_vid_axes: plt.Axes = None

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fps = 20.0
        frameSize = self.env.field_size.astype(int)
        self.sol_vid_writer = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=fps, frameSize=frameSize)
        _, self.sol_vid_axes = plt.subplots(figsize=(10, 10))
        self.sol_vid_axes.set_position([0.01, 0.01, 0.98, 0.98])

    def frame_network_vid(self, network: Network, text: str = None, show_frame: bool = False, frame_count: int = 1):
        if self.sol_vid_writer is None:
            # raise Exception("Solution Video generation must first be initialized using init_solution_vid")
            return

        try:
            surface = self.sol.drawer.draw_network(network, draw_env=True, draw_footprint=True, draw_id=True, draw_forces=True)
            if text is not None:
                s_txt = FONT.render(text, True, "red")
                surface.blit(s_txt, (10, self.env.field_size[1] - FONTSIZE - 1))

            frame = pg.surfarray.pixels3d(surface).transpose((1, 0, 2))
            if frame_count == 1:
                self.sol_vid_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                for _ in range(frame_count):
                    self.sol_vid_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if show_frame:
                self.sol_vid_axes.imshow(frame)
                plt.pause(0.001)
                plt.show(block=False)
                plt.pause(0.001)
        except Exception as e:
            LOGGER.log_error("Exception during frame drawing", e)

    def close_network_vid(self):
        del self.sol_vid_writer
        self.sol_vid_writer = None


class SequenceDrawer:
    def __init__(self, seq: Sequence):
        self.seq = seq
        self.sol = seq.sol
        self.env = self.sol.env
        self.sce = seq.sce

    def save_sequence_vid(self, filename: str = './output/controller_sequence.avi', show_graph: bool = True, frame_length: float = 1/20):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        frameSize = self.env.field_size.astype(int)
        sequence_vid_writer = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=1/frame_length, frameSize=frameSize)
        seq_surface_bg = self.env.drawer.draw_environment(copy=True)
        if show_graph:
            seq_surface_bg.blit(self.sol.drawer.draw_graph(self.sol.graph, draw_env=True, draw_id=True), (0, 0))

        for time in np.arange(self.sce.time_max, step=frame_length):
            seq_surface = seq_surface_bg.copy()
            seq_surface.blit(self.draw_sequence(time=time, draw_env=False, draw_item=True, draw_id=True, draw_state=True), (0, 0))

            s_txt = FONT.render("{0:.2f}".format(time), True, "red")
            seq_surface.blit(s_txt, (10, self.env.field_size[1] - FONTSIZE - 1))

            frame = pg.surfarray.pixels3d(seq_surface).transpose((1, 0, 2))
            sequence_vid_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        del sequence_vid_writer

    def clean_surfaces(self):
        self.sol.drawer.clean_surfaces()

    def draw_sequence(self, time: float, draw_env: bool = True, draw_item: bool = True, draw_station_items: bool = True, draw_id: bool = True, draw_state: bool = True) -> pg.Surface:
        if self.seq is None:
            return
        surface = pg.Surface(self.env.field_size, pg.SRCALPHA)
        if draw_env:
            surface.blit(self.env.drawer.draw_environment(), (0, 0))

        for mover in self.seq.movers:
            self._draw_mover(surface=surface, mover=mover, time=time, draw_item=draw_item, draw_id=draw_id, draw_state=draw_state)

        if draw_station_items:
            self._draw_sequence_stations(surface=surface, time=time)

        return surface

    def _draw_mover(self, surface: pg.Surface, mover: Mover, time: float, draw_item: bool, draw_id: bool, draw_state: bool) -> tuple[pg.Surface, pg.Rect]:
        pos = mover.get_pos(time) - self.env.mover_size / 2

        pg.draw.rect(surface, C_MOVER, (*pos, *self.env.mover_size), border_radius=10)
        if draw_item:
            item = mover.get_item(time)
            if item is None or item.is_empty:
                pg.draw.rect(surface, C_ITEMS[0], (pos[0] + self.env.mover_size[0] * 0.25, pos[1] + self.env.mover_size[1] * 0.25, *self.env.mover_size * 0.5), border_radius=5)
            else:
                pg.draw.rect(surface, C_ITEMS[item], (pos[0] + self.env.mover_size[0] * 0.25, pos[1] + self.env.mover_size[1] * 0.25, *self.env.mover_size * 0.5), border_radius=5)
        if draw_id or draw_state:
            text = ""
            if draw_id:
                text += "{0:2d}: ".format(mover.id) if mover.id is not None else "--: "
            if draw_state:
                state = mover.get_state(time)
                text += state.name if state else "NO_STATE"

            s_t = FONT.render(text, True, C_TEXT)
            surface.blit(s_t, (pos[0] + 10, pos[1] + self.env.mover_size[1] - 20))

    def _draw_sequence_stations(self, surface: pg.Surface, time: float, n: int = 10, d_min: float = 1.1, shuffle_provider: bool = False):
        if self.seq is None:
            return
        item_size = self.env.mover_size * 0.5
        conv_size = [n*item_size[0]*d_min, item_size[1]*1.2]

        for provider in self.env.providers:
            if provider.pos[0] < provider.size[0]:
                dir = np.array([-1, 0])
                x = (provider.pos[0]-n*item_size[0]*d_min-provider.size[0]*0.5, provider.pos[1]-item_size[1]*0.6, *conv_size)
                pg.draw.rect(surface, C_FORBIDDEN, x)
            elif provider.pos[0] > self.env.field_size[0] - provider.size[0]:
                dir = np.array([1, 0])
            elif provider.pos[1] < provider.size[1]:
                dir = np.array([0, -1])
            else:
                dir = np.array([0, 1])

            events: list[Event] = None
            events = self.seq.events[provider]
            if not events or events[-1].time_start <= time:
                continue
            idx = next(i for i in range(len(events)) if events[i].time_completes > time)
            events_shown = events[idx:min(idx+n, len(events))]

            l_item = np.linalg.norm(item_size * dir, ord=1) * d_min
            if provider.reset != 0:
                v_item = l_item / provider.reset * d_min
            else:
                v_item = l_item / min(en.time_start - e.time_start for e, en in zip(events[:-1], events[1:])) * d_min
            dpos = max(0, min(l_item, (events_shown[0].time_completes - time) * v_item))

            for i, event in enumerate(events_shown):
                if shuffle_provider:
                    pos = provider.pos - item_size * 0.5 + (dpos + i * l_item) * dir
                else:
                    pos = provider.pos - item_size * 0.5 - (time - event.time_completes) * dir * v_item
                pg.draw.rect(surface, C_ITEMS[event.item], (*pos, *item_size), border_radius=5)

        for receiver in self.env.receivers:
            if receiver.pos[0] < receiver.size[0]:
                dir = np.array([-1, 0])
            elif receiver.pos[0] > self.env.field_size[0] - receiver.size[0]:
                dir = np.array([1, 0])
                pg.draw.rect(surface, C_FORBIDDEN, (receiver.pos[0]+receiver.size[0]*0.5, receiver.pos[1]-item_size[1]*0.6, *conv_size))
            elif receiver.pos[1] < receiver.size[1]:
                dir = np.array([0, -1])
            else:
                dir = np.array([0, 1])

            events: list[Event] = None
            events = self.seq.events[receiver]
            if not events or events[0].time_completes > time:
                continue
            if events[-1].time_completes <= time:
                idx = len(events) - 1
            else:
                idx = next(i for i in range(len(events)) if events[i].time_completes <= time and events[i+1].time_completes > time)
            events_shown = events[max(0, idx-n):idx+1]

            l_item = np.linalg.norm(item_size * dir, ord=1) * d_min
            if receiver.reset != 0:
                v_item = l_item / receiver.reset * d_min
            else:
                v_item = l_item / min((en.time_start - e.time_start for e, en in zip(events[:-1], events[1:])), default=1) * d_min
            for event in events_shown:
                pos = receiver.pos - item_size * 0.5 + (time - event.time_completes) * dir * v_item
                pg.draw.rect(surface, C_ITEMS[event.item], (*pos, *item_size), border_radius=5)
