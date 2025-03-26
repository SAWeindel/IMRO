from __future__ import annotations
from tools.draw import FORCE_DICT
from definition import *
from typing import Hashable
from dataclasses import dataclass, Field, field
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # noqa: E402
import pygame as pg


C_BACKGROUND = "white"
C_TEXT = "black"
C_LINES = "grey20"
C_UI = "grey90"
C_BUTTON = "grey70"
C_BUTTON_ACTIVE = "grey50"
C_BUTTON_OFF = "grey30"
C_HIGHLIGHT = pg.Color(255, 255, 0, 100)

FONTSIZE = 12
LEADING = 3
SPACING = FONTSIZE + LEADING


@dataclass
class Button:
    text: str
    pos: pg.Vector2
    size: pg.Vector2
    clickable: bool = True
    pressed: bool = False
    on_press: callable[bool] = None
    statedict: dict = None
    state: Hashable = None
    stateval: Field = None
    statenone: bool = False
    toggle: bool = False
    d = None

    def test_pressed(self, pos: tuple[float, float], pressed: bool) -> bool:
        if self.d[1].collidepoint(pos):
            if not self.clickable:
                return True
            if not self.toggle:
                self.press(pressed)
            elif pressed:
                self.press()
            return True
        return False

    def press(self, pressed: bool = None):
        pressed = pressed if pressed is not None else not self.pressed
        if self.on_press:
            self.pressed = pressed
            self.on_press(pressed)
        elif self.stateval is None:
            self.statedict[self.state] = pressed
        elif pressed:
            if self.statenone and self.statedict[self.state] == self.stateval:
                self.statedict[self.state] = None
            else:
                self.statedict[self.state] = self.stateval

    def draw(self) -> tuple[pg.Surface, pg.Rect]:
        if not self.on_press and not self.stateval:
            self.pressed = self.statedict[self.state]

        surface = pg.Surface(self.size, pg.SRCALPHA)
        if not self.clickable:
            c = C_BUTTON_OFF
        elif self.toggle and self.pressed:
            c = C_BUTTON_ACTIVE
        else:
            c = C_BUTTON
        surface.fill(c)
        pg.draw.rect(surface, C_LINES, (0, 0, *self.size - [1, 1]), width=2)
        text = pg.font.SysFont("consolas", size=FONTSIZE).render(self.text, True, C_TEXT)
        r = text.get_rect()
        r = r.move((self.size - (r.width, r.height)) / 2)
        surface.blit(text, r)
        rect = surface.get_rect().move(self.pos)
        self.d = (surface, rect)
        return self.d


class GUI:
    def __init__(self, env: Environment = None, sol: Solution = None, seq: Sequence = None):
        self.env: Environment = None
        self.sol: Solution = None
        self.sce: Scenario = None
        self.seq: Sequence = None

        pg.init()
        pg.display.set_caption('Simulation GUI')
        icon = pg.Surface((1, 1))
        icon.fill("white")
        pg.display.set_icon(icon)

        pg.font.init()
        self.font = pg.font.SysFont("consolas", size=FONTSIZE)

        self.quit_after_close: bool = True  # set false for Nupyter Notebook

        self.states = {
            "state_playing": False,
            "state_cont_fw": False,
            "state_cont_bw": False,
            "state_scroll_drag": False,
            "state_draw_seq": False,
            "type_view_overlay": None,
            "state_log_active": None,
            "state_log_desired": "env",
        }

        self.ui_changed = True
        self.view_changed = True
        self.log_changed = True

        if seq:
            self.load_sequence(seq=seq)
        elif sol:
            self.load_solution(sol=sol)
        elif env:
            self.load_environment(env=env)
        else:
            raise Exception("Either Environment, Solution or Sequence must be specified")

        self.generate_visuals()

    def load_environment(self, env: Environment):
        self.env = env

        self.view_size_unscaled = self.env.field_size
        self.view_scale = 1000 / np.max(self.env.field_size)
        self.view_size = self.view_size_unscaled * self.view_scale
        self.view_surface = pg.Surface(self.view_size, pg.SRCALPHA)

        self.log_size = pg.math.Vector2(1200, 1000)
        self.log_surface = pg.Surface(self.log_size, pg.SRCALPHA)

        self.ui_height = 150

        self.screen_ref_size = self.view_size
        screen_width = self.screen_ref_size[0] + self.log_size[0] + 150
        screen_height = max(self.screen_ref_size[1], self.log_size[1]) + self.ui_height + 100
        self.screen_unscaled_size = pg.math.Vector2(screen_width, screen_height)
        self.screen_unscaled: pg.Surface = None
        self.screen_size: pg.math.Vector2 = None
        self.screen: pg.Surface = None

        self.view_offset = pg.math.Vector2(50, (self.screen_unscaled_size[1] - self.screen_ref_size[1] - 200) / 2)
        self.log_offset = pg.math.Vector2(100 + self.screen_ref_size[0], 25)
        self.ui_offset = pg.math.Vector2(50, self.screen_unscaled_size[1] - self.ui_height - 50)

        self.view_surface_rect = self.view_surface.get_rect().move(self.view_offset)
        self.log_surface_rect = self.log_surface.get_rect().move(self.log_offset)

        self.view_environment_surface: pg.Surface = None
        self.view_flow_surface: pg.Surface = None
        self.view_network_surfaces: list[pg.Surface] = None
        self.view_current_network: int = None
        self.view_graph_surface: pg.Surface = None

        self.ui_num_view_buttons = 5
        self.ui_num_log_buttons = 5
        self.ui_buttons: dict[Button] = None

        self.log_texts: dict[str] = {}
        self.log_scroll_px = 0
        self.log_scroll_px_per_line = 0
        self.log_scroll_pos = 0
        self.log_scroll_max = 0
        self.log_max_lines = int(np.floor((self.log_size[1] - 60) / SPACING))

        self.log_actions_idx = []

        self.log_scroll_box_height = self.log_size[1] - 40
        self.log_scroll_mover_height = self.log_size[1] - 40
        self.log_scroll_box_offset = pg.math.Vector2(self.log_size[0] - 40, 20)
        self.log_scroll_mover_offset = pg.math.Vector2(self.log_size[0] - 40, 20)
        self.log_scroll_box: pg.Surface = None
        self.log_scroll_box_rect: pg.Rect
        self.log_scroll_mover: pg.Surface = None

        self.clock = pg.time.Clock()
        self.time = 0.0
        self.frame_length = 1/20
        self.step_length = 0.2
        self.playback_speed = 1

    def load_solution(self, sol: Solution):
        self.sol = sol
        self.load_environment(env=self.sol.env)

        self.states["draw_flow"] = True
        self.states["type_view_overlay"] = "network"
        self.states["state_log_desired"] = "network"

    def load_sequence(self, seq: Sequence):
        self.seq = seq
        self.sce = seq.sce
        self.load_solution(sol=self.seq.sol)

        self.states["state_draw_seq"] = True
        self.states["state_playing"] = True
        self.states["type_view_overlay"] = "graph"
        self.states["state_log_desired"] = "sequence"

    def generate_visuals(self):
        has_sol = (self.sol is not None)
        has_seq = (self.seq is not None)

        self.ui_buttons = {}
        size = pg.math.Vector2(int(self.screen_ref_size[0] / self.ui_num_view_buttons) - 10, int(self.ui_height / 2) - 10)

        pos = [pg.math.Vector2(self.ui_offset[0] + 5 + i * int(self.screen_ref_size[0] / self.ui_num_view_buttons), self.ui_offset[1] + 5) for i in range(6)]
        self.ui_buttons["draw_flow"] = Button(text="Flow", pos=pos[0], size=size, clickable=has_sol, toggle=False, statedict=self.states, state="type_view_overlay", stateval="flow", statenone=True)
        self.ui_buttons["network_bw"] = Button(text="Previous Network", pos=pos[1], size=size, clickable=has_sol, toggle=False, on_press=self.action_network_bw)
        self.ui_buttons["network_fw"] = Button(text="Next Network", pos=pos[2], size=size, clickable=has_sol, toggle=False, on_press=self.action_network_fw)
        self.ui_buttons["draw_graph"] = Button(text="Graph", pos=pos[3], size=size, clickable=has_sol, toggle=False, statedict=self.states, state="type_view_overlay", stateval="graph", statenone=True)
        self.ui_buttons["state_draw_seq"] = Button(text="Sequence", pos=pos[4], size=size, clickable=has_seq, toggle=True, statedict=self.states, state="state_draw_seq")

        size = pg.math.Vector2(int(self.screen_ref_size[0] / 3) - 10, int(self.ui_height / 2) - 10)
        pos = pg.math.Vector2(self.ui_offset[0] + 5 + int(self.screen_ref_size[0] / 3), self.ui_offset[1] + int(self.ui_height / 2) + 5)
        btn = Button(text="Play Animation", pos=pos, size=size, clickable=has_seq, toggle=True, statedict=self.states, state="state_playing")
        self.ui_buttons["state_playing"] = btn

        for i, (state, text) in enumerate((("state_cont_bw", "Step Backward"),
                                           ("state_cont_fw", "Step Forward"))):
            pos = pg.math.Vector2(self.ui_offset[0] + 5 + 2 * i * int(self.screen_ref_size[0] / 3), self.ui_offset[1] + int(self.ui_height / 2) + 5)
            btn = Button(text=text, pos=pos, size=size, clickable=has_seq, toggle=False, statedict=self.states, state=state)
            self.ui_buttons[state] = btn

        size = pg.math.Vector2(int(self.log_size[0] / self.ui_num_view_buttons) - 10, int(self.ui_height / 2) - 10)
        for i, (state, text, clk, val) in enumerate((("log_env", "Environment", True, "env"),
                                                     ("log_network", "Networks", has_sol, "network"),
                                                     ("log_graph", "Graph", has_sol, "graph"),
                                                     ("log_actions", "Actions", has_seq, "actions"),
                                                     ("log_sequence", "Sequence", has_seq, "sequence"))):
            pos = pg.math.Vector2(self.log_offset[0] + 5 + i * int(self.log_size[0] / self.ui_num_log_buttons), self.ui_offset[1] + int(self.ui_height / 2) + 5)
            btn = Button(text=text, pos=pos, size=size, clickable=clk, toggle=False, statedict=self.states, state="state_log_desired", stateval=val)
            self.ui_buttons[state] = btn

        self.log_scroll_box = pg.Surface((20, self.log_scroll_box_height), pg.SRCALPHA)
        self.log_scroll_box.fill(C_BUTTON)
        self.log_scroll_box_rect = self.log_scroll_box.get_rect().move(self.log_offset + self.log_scroll_box_offset)

        self.view_environment_surface = self.env.drawer.draw_environment()

        if has_sol:
            self.view_flow_surface = self.sol.drawer.draw_network(network=self.sol.flow, draw_footprint=False)
            self.view_network_surfaces = []
            for network in self.sol.networks:
                self.view_network_surfaces.append(self.sol.drawer.draw_network(network=network))
            self.view_current_network = len(self.view_network_surfaces) - 1
            self.view_graph_surface = self.sol.drawer.draw_graph(graph=self.sol.graph)

    def run(self, quit_after_close: bool = True):
        self.quit_after_close = quit_after_close
        self.screen_unscaled = pg.Surface(self.screen_unscaled_size, pg.SRCALPHA)
        self.screen_unscaled.fill(C_BACKGROUND)
        pg.draw.rect(self.screen_unscaled, C_LINES, (self.view_offset[0] - 2, self.view_offset[1] - 2, self.screen_ref_size[0] + 5, self.screen_ref_size[1] + 5), width=3)

        info = pg.display.Info()
        maxsize = pg.math.Vector2([info.current_w, info.current_h]) * 0.8
        self.screen_size = self.screen_unscaled_size * min(maxsize.elementwise() / self.screen_unscaled_size.elementwise())
        self.screen = pg.display.set_mode(self.screen_size, flags=pg.RESIZABLE)
        pg.transform.smoothscale(self.screen_unscaled, self.screen_size, self.screen)

        self.running = True
        while self.running:
            self.handle_events()

            if self.states["state_cont_fw"]:
                self.action_cont_fw()
            elif self.states["state_cont_bw"]:
                self.action_cont_bw()
            elif self.states["state_playing"]:
                self.action_frame_fw()
            else:
                self.action_draw()

    def event_scale(self, pos: tuple[int, int]) -> tuple[float, float]:
        return pos * self.screen_unscaled_size.elementwise() / self.screen_size.elementwise()

    def flip_display(self):
        pg.transform.smoothscale(self.screen_unscaled, self.screen_size, self.screen)
        pg.display.flip()

    def handle_events(self):
        for event in pg.event.get():
            try:
                match event.type:
                    case pg.QUIT:
                        self.running = False
                        self.action_quit()
                    case pg.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            pos = self.event_scale(pg.mouse.get_pos())
                            for btn in self.ui_buttons.values():
                                if btn.test_pressed(pos=pos, pressed=True):
                                    break
                            if self.log_scroll_box_rect.collidepoint(pos):
                                self.states["state_scroll_drag"] = True
                    case pg.MOUSEBUTTONUP:
                        if event.button == 1:
                            pos = self.event_scale(pg.mouse.get_pos())
                            for btn in self.ui_buttons.values():
                                if btn.test_pressed(pos=pos, pressed=False):
                                    break
                    case pg.MOUSEWHEEL:
                        pos = self.event_scale(pg.mouse.get_pos())
                        if self.view_surface_rect.collidepoint(pos):
                            if event.y > 0:
                                self.action_step_bw()
                            elif event.y < 0:
                                self.action_step_fw()
                        elif self.log_surface_rect.collidepoint(pos):
                            self.action_scroll_log(event.y * 3)
                    case pg.KEYDOWN:
                        match event.key:
                            case pg.K_SPACE: self.states["state_playing"] = not self.states["state_playing"]
                            case pg.K_LEFT | pg.K_a: self.states["state_cont_bw"] = True
                            case pg.K_RIGHT | pg.K_d: self.states["state_cont_fw"] = True
                    case pg.KEYUP:
                        match event.key:
                            case pg.K_LEFT | pg.K_a: self.states["state_cont_bw"] = False
                            case pg.K_RIGHT | pg.K_d: self.states["state_cont_fw"] = False
                    case pg.VIDEORESIZE:
                        self.screen_size = pg.math.Vector2(event.w, event.h)
                        self.screen = pg.display.set_mode(self.screen_size, flags=pg.RESIZABLE)
                        self.flip_display()
            except Exception as e:
                print("Exception during event handling", event, e)

            self.ui_changed = True
            self.view_changed = True
            self.log_changed = True

        if self.states["state_scroll_drag"]:
            if not pg.mouse.get_pressed()[0]:
                self.states["state_scroll_drag"] = False
            elif self.log_scroll_max > 0:
                pos = self.event_scale(pg.mouse.get_pos())[1]
                line = (pos - self.log_offset[1] - self.log_scroll_box_offset[1] - self.log_scroll_mover_height / 2) \
                    / (self.log_scroll_box_height - self.log_scroll_mover_height) * self.log_scroll_max
                self.action_scroll_log_to(int(np.round(line)))

    def action_draw(self):
        if self.ui_changed:
            self.draw_ui()
        if self.view_changed:
            self.draw_view(time=self.time)
        if self.log_changed:
            self.draw_log(time=self.time)

        if self.ui_changed or self.view_changed or self.log_changed:
            self.flip_display()
            self.ui_changed = self.view_changed = self.log_changed = False

        dt = self.clock.tick(1 / self.frame_length)
        fps = self.clock.get_fps()
        if dt > self.frame_length * 1000 * 1.1 and fps > 0 and 1.1 * fps < 1 / self.frame_length:
            print("Timestep took too long: %d ms. Ten frame average is %d ms." % (dt, 1000 / fps))

    def action_quit(self):
        if self.quit_after_close:
            pg.font.quit()
            pg.quit()
            quit()
        else:
            self.screen = pg.display.set_mode(self.screen_size, flags=pg.HIDDEN | pg.RESIZABLE)

    def action_cont_fw(self):
        self.states["state_playing"] = False
        target = np.floor(np.round(((self.time + self.step_length) / self.step_length), 1)) * self.step_length
        if target > self.sce.time_max - 2 * self.step_length:
            target = 0.0

        while self.time < target:
            self.view_changed = True
            self.log_changed = True
            self.action_draw()
            self.time += self.frame_length * self.playback_speed * 3
        self.time = target

    def action_cont_bw(self):
        self.states["state_playing"] = False
        target = np.ceil(np.round(((self.time - self.step_length) / self.step_length), 1)) * self.step_length
        if target < 0:
            target = self.sce.time_max

        while self.time > target:
            self.view_changed = True
            self.log_changed = True
            self.action_draw()
            self.time -= self.frame_length * self.playback_speed * 3
        self.time = target

    def action_step_fw(self):
        self.states["state_playing"] = False
        self.time = np.floor(np.round(((self.time + self.step_length) / self.step_length), 1)) * self.step_length
        if self.time > self.sce.time_max:
            self.time = 0.0
        self.view_changed = True
        self.log_changed = True
        self.action_draw()

    def action_step_bw(self):
        self.states["state_playing"] = False
        self.time = np.ceil(np.round(((self.time - self.step_length) / self.step_length), 1)) * self.step_length
        if self.time < 0:
            self.time = self.sce.time_max
        self.view_changed = True
        self.log_changed = True
        self.action_draw()

    def action_frame_fw(self):
        self.view_changed = True
        self.log_changed = True
        self.action_draw()
        if self.states["state_playing"]:
            self.time += self.frame_length * self.playback_speed
        if self.time > self.sce.time_max:
            self.time = 0.0

    def action_network_bw(self, pressed: bool):
        if not pressed:
            return
        if self.states["type_view_overlay"] != "network":
            self.states["type_view_overlay"] = "network"
            self.view_current_network = 0
        else:
            self.view_current_network = max(self.view_current_network - 1, 0)
        self.states["state_log_desired"] = "network"
        self.log_texts["network"] = self.generate_text_log("network")
        self.view_changed = True
        self.log_changed = True

    def action_network_fw(self, pressed: bool):
        if not pressed:
            return
        if self.states["type_view_overlay"] != "network":
            self.states["type_view_overlay"] = "network"
            self.view_current_network = len(self.view_network_surfaces) - 1
        else:
            self.view_current_network = min(self.view_current_network + 1, len(self.view_network_surfaces) - 1)
        self.states["state_log_desired"] = "network"
        self.log_texts["network"] = self.generate_text_log("network")
        self.view_changed = True
        self.log_changed = True

    def action_scroll_log(self, dir: int):
        self.action_scroll_log_to(self.log_scroll_pos - dir)

    def action_scroll_log_to(self, pos: int):
        self.log_scroll_pos = pos
        if self.log_scroll_pos < 0:
            self.log_scroll_pos = 0
        if self.log_scroll_pos > self.log_scroll_max:
            self.log_scroll_pos = self.log_scroll_max
        if self.log_scroll_max == 0:
            self.log_scroll_mover_offset = pg.math.Vector2(self.log_size[0] - 40, 20)
        else:
            self.log_scroll_mover_offset = pg.math.Vector2(self.log_size[0] - 40, 20 + (self.log_scroll_box_height - self.log_scroll_mover_height) * (self.log_scroll_pos / self.log_scroll_max))
        self.log_changed = True

    def draw_ui(self):
        self.screen_unscaled.blits(btn.draw() for btn in self.ui_buttons.values())

    def draw_log(self, time: float):
        if self.states["state_log_active"] != self.states["state_log_desired"]:
            self.action_scroll_log_to(0)
            self.states["state_log_active"] = self.states["state_log_desired"]

        match self.states["state_log_desired"]:
            case "sequence":
                text = ""
                for mover in self.seq.movers:
                    pos = mover.get_pos(time)
                    state = mover.get_state(time)
                    state = state.name if state else "NO_STATE"
                    text += "Mover {0:2d}: [{1:4.0f},{2:4.0f}], {3:11s} doing {4}\n".format(mover.id, pos[0], pos[1], state, mover.get_action(time))
                log_text = text.splitlines()
                highlight = []
            case "actions":
                if self.states["state_log_desired"] not in self.log_texts:
                    self.log_texts[self.states["state_log_desired"]] = self.generate_text_log(self.states["state_log_desired"])
                log_text = self.log_texts[self.states["state_log_desired"]]
                highlight = []
                for idx, m in zip(self.log_actions_idx, self.seq.movers):
                    a = m.get_action_index(time)
                    if a is not None:
                        highlight.append(idx + a)
            case _:
                if self.states["state_log_desired"] not in self.log_texts:
                    self.log_texts[self.states["state_log_desired"]] = self.generate_text_log(self.states["state_log_desired"])
                log_text = self.log_texts[self.states["state_log_desired"]]
                highlight = []

        if self.seq is not None:
            header_text = "Displaying Time {0:6.2f} - Step {1:3d}".format(self.time, int(self.time * self.step_length))
        elif self.sol is not None:
            header_text = "No Sequence Loaded"
        else:
            header_text = "No Solution Loaded"

        self.log_scroll_max = max(0, len(log_text) - self.log_max_lines)
        self.log_scroll_mover = pg.Surface((20, self.log_scroll_box_height * min(1, self.log_max_lines / len(log_text))), pg.SRCALPHA)
        self.log_scroll_mover_height = self.log_scroll_mover.get_size()[1]
        self.log_scroll_mover.fill(C_BUTTON_ACTIVE)

        self.log_surface.fill(C_UI)
        pg.draw.rect(self.log_surface, C_LINES, (0, 0, *self.log_size - [1, 1]), width=2)
        self.log_surface.blit(self.font.render(header_text, True, C_TEXT), (20, 20))
        self.log_surface.blit(self.draw_text(log_text, highlight=highlight, max_len=150, linebreak=("\\", "   ")), (20, 20 + 1.5 * SPACING))

        self.log_surface.blit(self.log_scroll_box, self.log_scroll_box_offset)
        self.log_surface.blit(self.log_scroll_mover, self.log_scroll_mover_offset)
        self.screen_unscaled.blit(self.log_surface, self.log_offset)

    def draw_view(self, time: float):
        view_surface_unscaled = self.view_environment_surface.copy()
        match self.states["type_view_overlay"]:
            case None:
                view_surface_unscaled = self.view_environment_surface.copy()
            case "flow":
                view_surface_unscaled = self.view_flow_surface.copy()
            case "network":
                view_surface_unscaled = self.view_network_surfaces[self.view_current_network].copy()
            case "graph":
                view_surface_unscaled = self.view_graph_surface.copy()

        if self.states["state_draw_seq"]:
            view_surface_unscaled.blit(self.seq.drawer.draw_sequence(time=time, draw_env=False), (0, 0))

        self.screen_unscaled.blit(pg.transform.smoothscale(view_surface_unscaled, self.view_size), self.view_surface_rect)

    def draw_text(self, lines: list[str], highlight: list[int] = [], max_len: int = None, linebreak: tuple[str, str] = ("\\", "")):
        lines = lines[self.log_scroll_pos:]
        highlight = [h - self.log_scroll_pos for h in highlight if h >= self.log_scroll_pos]
        max_width = 0
        surfaces = []
        if max_len:
            newlines = []
            newhighlight = []
            for (i, line) in enumerate(lines):
                if len(line) < max_len:
                    newlines.append(line)
                    if i in highlight:
                        newhighlight.append(len(newlines) - 1)
                else:
                    count = 1
                    split = line.split(",") + ['']
                    split_char = ','
                    if len(split) == 2 or any(len(s) > max_len for s in split):  # no ',' in line
                        split = line.split(" ") + ['']
                        split_char = ' '
                    newline = ""
                    for seg, next in zip(split, split[1:]):
                        newline += seg + split_char
                        if len(newline) + len(next) > max_len:
                            newlines.append(newline + linebreak[0])
                            newline = linebreak[1]
                            count += 1
                    newlines.append(newline[:-1])

                    if i in highlight:
                        newhighlight.extend(range(len(newlines) - count, len(newlines)))

                if len(newlines) > self.log_max_lines:
                    break

            highlight = newhighlight
            lines = newlines

        for i in range(min(len(lines), self.log_max_lines)):
            s = self.font.render(lines[i], True, C_TEXT, C_HIGHLIGHT if i in highlight else None)
            r = s.get_rect().move(0, i * SPACING)
            max_width = max(max_width, r.width)
            surfaces.append((s, r))

        surface = pg.Surface([max_width, len(lines) * SPACING - LEADING], pg.SRCALPHA)
        surface.blits(surfaces)

        return surface

    def generate_text_log(self, name: str) -> list[str]:
        match name:
            case "env":
                lines = ["Environment '{0}': {1:d} Providers, {2:d} Receivers, Items {3}"
                         .format(self.env.name, len(self.env.providers), len(self.env.receivers), self.env.items)]
                lines.extend([repr(station) for station in self.env.providers])
                lines.extend([repr(station) for station in self.env.receivers])
                return lines
            case "graph":
                lines = []
                for node in self.sol.graph.nodes:
                    lines.append("Node {0} at [{1:4.0f},{2:4.0f}], {3} edges, {4} incoming, time_clear {5:4.2f}, station is {6}".
                                 format("{0:2d}".format(node.id) if node.id is not None else 'N ',
                                        node.pos[0], node.pos[1], len(node.edges), len(node.edges_incoming), node.time_clear, node.station))
                    if node.station:
                        lines.append("Station {0}".format(node.station))
                    if not node.can_wait:
                        lines.append("Not Awaitable")
                    if node.can_skip:
                        lines.append("Skippable")
                    # if node.is_queued: lines.append("Queued")
                    lines.append("Edges:")
                    lines.extend(["    {0}".format(repr(edge)) for edge in node.edges])
                    lines.append("Edges incoming:")
                    lines.extend(["    {0}".format(repr(edge)) for edge in node.edges_incoming])
                    lines.append("")
                return lines
            case "actions":
                lines = []
                for mover in self.seq.movers:
                    text = ""
                    text += "{0}:\n".format(repr(mover))
                    for (i, action) in enumerate(mover.actions):
                        text += "    {0:3d}: {1}\n".format(i, repr(action))

                    for i, (time, state) in enumerate(mover.states):
                        text += "    {0:3d}: {1:6.2f} {2:11s}\n".format(i, time, state)

                    self.log_actions_idx.append(len(lines) + 1)
                    lines.extend(text.splitlines())
                return lines
            case "network":
                network = self.sol.networks[self.view_current_network]
                network_num = (int)(self.view_current_network / 2) + 1
                network_maxnum = (int)(len(self.view_network_surfaces) / 2) + 1
                network_preforce = self.view_current_network % 2 == 0
                lines = ["Displaying network {0}/{1} {2}".format(network_num, network_maxnum,
                         "before force movement" if network_preforce else "after force movement")]
                txt = ""
                for c, v in FORCE_DICT.items():
                    txt += "{0}: {1}, ".format(c, v)
                lines.append(txt)
                for prim in network.primitives:
                    lines.append("{0}".format(repr(prim)))
                    if prim.incoming:
                        lines.append("    Incoming:")
                        lines.extend(["        {0}".format(repr(arc)) for arc in prim.incoming])
                    if prim.outgoing:
                        lines.append("    Outgoing:")
                        lines.extend(["        {0}".format(repr(arc)) for arc in prim.outgoing])
                    if network.forces_comp and prim in network.forces_comp:
                        txt = ""
                        for n, v in network.forces_comp[prim]:
                            txt += "{0}: {1}, ".format(n, np.round(v, 2))
                        if len(txt) > 0:
                            lines.append("    Forces:")
                            lines.append("        {0}".format(txt[:-2]))
                    lines.append("")

                if network.sections is not None:
                    lines.append("Sections:")
                    for section in network.sections.values():
                        lines.append("    {0}".format(repr(section)))
                    lines.append("")

                if network.paths is not None:
                    lines.append("Paths:")
                    for path in network.paths.values():
                        lines.append("    {0}".format(repr(path)))
                    lines.append("")

                for arc in network.arcs:
                    lines.append("Arc {0}".format(arc))
                    if network.torques_comp and arc in network.torques_comp:
                        txt = ""
                        for n, v in network.torques_comp[arc]:
                            txt += "{0}: {1}, ".format(n, np.round(v, 2))
                        if len(txt) > 0:
                            lines.append("    Torques:")
                            lines.append("        {0}".format(txt[:-2]))
                    lines.append("")

                return lines[:-1]

            case _:
                return []
