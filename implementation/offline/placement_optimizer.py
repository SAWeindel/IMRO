from __future__ import annotations
from tools.log import LOGGER
from tools.draw import OptimizerDrawer
from definition import *
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from collections import deque
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)


class Stage(Enum):
    FIX_POS = 0
    FIX_ROT = 1
    FIX_SHIFT = 2
    FINAL = 3


@dataclass
class SimStep:
    step: int = np.nan
    stage: int = np.nan
    k_step: float = np.nan
    num_violation: int = np.nan
    avg_move: float = np.nan
    max_move: float = np.nan
    avg_vel: float = np.nan
    max_vel: float = np.nan
    avg_force: float = np.nan
    max_force: float = np.nan
    max_vel_exp: float = np.nan
    max_vel_block: float = np.nan
    max_pos_block: float = np.nan
    max_move_diff: float = np.nan
    avg_force_primarcs: float = np.nan
    max_force_primarcs: float = np.nan
    avg_force_primprims: float = np.nan
    max_force_primprims: float = np.nan
    avg_force_arcs_repr: float = np.nan
    max_force_arcs_repr: float = np.nan
    avg_force_arcs_step: float = np.nan
    max_force_arcs_step: float = np.nan
    avg_force_torque: float = np.nan
    max_force_torque: float = np.nan
    avg_torque: float = np.nan
    max_torque: float = np.nan
    score_primarc: float = np.nan
    score_primarc_abs: float = np.nan
    score_overlap: float = np.nan
    score_overlap_cutoff: float = np.nan
    score_violation: float = np.nan
    score_straighten: float = np.nan


class PlacementOptimizer:
    def __init__(self, env: Environment, sol: Solution):
        self.env = env
        self.sol = sol
        self.drawer = OptimizerDrawer(sol=self.sol)

        self.network: Network = None
        self.n_movable: int = None

        self.collision_gridlen: float = None
        self.collision_griddims: np.ndarray[int] = None
        self.collision_grid: np.ndarray[set[Primitive]] = None

        self.forces: dict[Primitive, np.ndarray[float]] = None
        self.forces_abs: dict[Primitive, float] = None
        self.forces_comp: dict[Primitive, list] = None
        self.torques: dict[Arc, float] = None
        self.torques_abs: dict[Arc, float] = None
        self.torques_comp: dict[Arc, list] = None
        self.velocities: dict[Primitive, np.ndarray[float]] = None

        self.vel_exp: dict[Primitive, np.ndarray[float]] = None
        self.vel_block: dict[Primitive, deque[np.ndarray[float]]] = None
        self.vel_block_vals: dict[Primitive, deque[np.ndarray[float]]] = None
        self.pos_block_vals: dict[Primitive, deque[np.ndarray[float]]] = None
        self.max_move_vals: dict[Primitive, deque[float]] = None
        self.k_step_vals: deque[float] = None

        self.max_violation: float = None
        self.max_force: float = None
        self.min_max_force: float = None
        self.max_vel: float = None
        self.max_move_diff: float = None

        self.overlap_primarcs: list[tuple[Arc, float]] = None
        self.overlap_primprims: list[tuple[Primitive, float, Primitive]] = None
        self.overlap_arclen: list[tuple[Arc, float]] = None
        self.violation_primangle: list[tuple[Primitive, Arc, Arc, float]] = None
        self.violation_primangle_straighten: list[tuple[Primitive, Arc, Arc, float]] = None

        self.score_directed_force: dict[Arc, float] = None
        self.score_primarc: dict[Arc, float] = None
        self.score_primarc_abs: dict[Arc, float] = None
        self.score_overlap_cutoff: dict[Arc, float] = None
        self.score_overlap: dict[Arc, float] = None
        self.score_violation: dict[Arc, float] = None

        self.score_straighten_oldarcs: dict[Arc, float] = None
        self.score_straighten: float = None

        self.step_log_list: list[SimStep] = None
        self.step_log: SimStep = None
        self.init_pos: dict[Primitive, np.ndarray[float]] = None
        self.old_max_straighten: float = np.inf

    def run(self, iter_max: int = 20, draw_stats: bool = True, show_stats: bool = False, draw_vid: bool = True, show_vid_freq: int | None = 0, checkpoint_path: str | None = "./output/"):
        step_max = [250, 250, 250, 250]
        n_init = [50, 50, 50, 50]
        k_g = [0.25, 0.2, 0.2, 0.05]
        k_frict = [0.8, 0.8, 0.8, 0.5]
        # violation_cutoff = [0.05, 0.025, 0.025, 0.001]
        violation_cutoff = [0.025, 0.025, 0.025, 0.001]
        force_cutoff = [np.inf, 5, 5, np.inf]
        vel_cutoff = [1, 0.5, 0.5, 0.5]
        # mdiff_cutoff = [2, 1, 1, 0.01]
        mdiff_cutoff = [5, 1, 1, 0.01]

        k_step_fct = [
            lambda step, n_init_iter, step_max_iter, k_g_iter: (pow(step / n_init_iter, 2) if step < n_init_iter else 1) * k_g_iter,
            lambda step, n_init_iter, step_max_iter, k_g_iter: (pow(step / n_init_iter, 2) if step < n_init_iter else 1) * k_g_iter,
            lambda step, n_init_iter, step_max_iter, k_g_iter: (pow(step / n_init_iter, 2) if step < n_init_iter else 1) * k_g_iter,
            lambda step, n_init_iter, step_max_iter, k_g_iter: np.exp(- step / step_max_iter) * k_g_iter
        ]

        force_score_config = [
            {'f_arcsteps': 0, 'straighten_corner': False},
            {'f_arcsteps': 0, 'min_angle': 89},
            {'min_angle': 89},
            {'force_ramp': 0.025, 'min_angle': 89, 'min_angle_straighten': 100}
        ]

        break_steps = [
            lambda init_complete, violation_below_cutoff, force_above_cutoff, vel_below_cutoff, move_diff_below_cutoff: (init_complete and (vel_below_cutoff or move_diff_below_cutoff)) or violation_below_cutoff,
            lambda init_complete, violation_below_cutoff, force_above_cutoff, vel_below_cutoff, move_diff_below_cutoff: (init_complete and (vel_below_cutoff or move_diff_below_cutoff)) or force_above_cutoff,
            lambda init_complete, violation_below_cutoff, force_above_cutoff, vel_below_cutoff, move_diff_below_cutoff: (init_complete and (vel_below_cutoff or move_diff_below_cutoff)) or force_above_cutoff,
            lambda init_complete, violation_below_cutoff, force_above_cutoff, vel_below_cutoff, move_diff_below_cutoff: init_complete and violation_below_cutoff
        ]

        self.network = self.sol.networks[-1]
        self.n_movable = self.network.n_movable
        self.init_pos = {p: p.pos.copy() for p in self.network.primitives}
        self.step_log_list = [SimStep()]
        stage = Stage.FIX_POS
        self.network.assign_ids()
        self.save_current_network()

        for iter in range(iter_max):
            step_max_iter = step_max[stage.value]
            k_g_iter = k_g[stage.value]
            k_frict_iter = k_frict[stage.value]
            violation_cutoff_iter = violation_cutoff[stage.value]
            force_cutoff_iter = force_cutoff[stage.value]
            vel_cutoff_iter = vel_cutoff[stage.value]
            n_init_iter = n_init[stage.value]
            mdiff_cutoff_iter = mdiff_cutoff[stage.value]
            force_score_config_iter = force_score_config[stage.value]
            k_step_fct_iter = k_step_fct[stage.value]
            break_steps_iter = break_steps[stage.value]
            break_steps_early = False

            LOGGER.log_empty()
            LOGGER.log_info("Iter {0}/{1}: {2}, max {3} steps:".format(iter + 1, iter_max, stage.name, step_max_iter))
            self.clear_iter()
            self.clear_step()
            for step in range(step_max_iter):
                self.step_log = SimStep(step=step, stage=stage.value)

                k_step_iter = k_step_fct_iter(step, n_init_iter, step_max_iter, k_g_iter)
                self.step(k_step=k_step_iter, step=step+1, k_frict=k_frict_iter, vel_cutoff=vel_cutoff_iter)
                # if step > 0: self.step(k_step=k_step_iter, step=step, k_frict=k_frict_iter, vel_cutoff=vel_cutoff_iter)

                self.get_forces_and_scores(**force_score_config_iter)

                self.step_log_list.append(self.step_log)
                if draw_vid:
                    self.drawer.frame_network_vid(network=self.network, text="Iter {0}: {1}, Step {2}".format(iter + 1, stage.name, step + 1), show_frame=show_vid_freq and step % show_vid_freq == 0)

                init_complete = step >= n_init_iter
                violation_below_cutoff = self.max_violation < violation_cutoff_iter
                force_above_cutoff = self.max_force > force_cutoff_iter
                vel_below_cutoff = self.max_vel < vel_cutoff_iter * k_step_iter
                move_diff_below_cutoff = self.max_move_diff < mdiff_cutoff_iter

                break_steps_early = break_steps_iter(init_complete, violation_below_cutoff, force_above_cutoff, vel_below_cutoff, move_diff_below_cutoff)

                if break_steps_early or (step > 0 and step % 10 == 0):
                    LOGGER.log_cont("Step {0:4d}, stepsize {1:4.3f}, overlap:{2:3d}, (average|max) move: {3:7.4f}|{4:7.4f}, vel: {5:7.4f}|{6:7.4f} (of {7:7.4f}), force: {8:7.4f}|{9:7.4f}"
                                    .format(self.step_log.step, k_step_iter, self.step_log.num_violation, self.step_log.avg_move, self.step_log.max_move, self.step_log.avg_vel, self.max_vel, vel_cutoff_iter * k_step_iter, self.step_log.avg_force, self.step_log.max_force))

                if step > 10:
                    max_force_abort = self.max_force > 10 * self.min_max_force
                    self.min_max_force = min(self.min_max_force, self.max_force)
                    if max_force_abort:
                        # LOGGER.log_warning("Force instability detected - aborting steps")
                        # print("Force instability detected - aborting steps")
                        # break
                        LOGGER.log_warning("Force instability detected - resetting velocity")
                        self.velocities = {p: np.zeros(2, float) for p in self.network.primitives}
                        self.min_max_force = np.inf

                if break_steps_early:
                    break

            LOGGER.log_info("Iter completed after {0}/{1} steps".format(step + 1, step_max_iter))
            self.log_scores(violation_cutoff=violation_cutoff_iter)
            if checkpoint_path:
                text = "Iter {0:2d} Optim Complete".format(iter + 1)
                name = "iter_{0:02}_a_pre_mod".format(iter + 1)
                self.sol.drawer.save_network_img(source=self.network, text=text, filename=checkpoint_path+name+".png", draw_env=True, draw_forces=True)
                LOGGER.to_json(self.network.to_dict(), path=checkpoint_path, name=name)

            if iter == iter_max - 1:
                LOGGER.log_info("Last iteration completed, stopping Iterations")
                break
            elif stage != Stage.FINAL and iter == iter_max - 2:
                LOGGER.log_info("Iter max reached, starting Finalizing")
                self.permute_remove_corners(max_angle=1, max_torque=np.inf)
                stage = Stage.FINAL

            match stage:
                case Stage.FIX_POS:
                    if not violation_below_cutoff:
                        self.permute_remove_corners()
                        self.permute_split_corners()
                        LOGGER.log_info("FIX_POS has not removed all violations, repeating stage")
                    else:
                        self.permute_remove_corners(max_angle=1)
                        LOGGER.log_info("No violations above threshold, starting next stage")
                        stage = Stage.FIX_ROT

                case Stage.FIX_ROT:
                    self.permute_remove_corners(max_angle=1, max_torque=np.inf)
                    if not break_steps_early:
                        LOGGER.log_info("Break condition not yet reached, repeating stage")
                    else:
                        LOGGER.log_info("Steps ended early, starting next stage")
                        stage = Stage.FIX_SHIFT

                case Stage.FIX_SHIFT:
                    self.permute_remove_corners(max_angle=1, max_torque=np.inf)
                    if not break_steps_early:
                        LOGGER.log_info("Break condition not yet reached, repeating stage")
                    else:
                        LOGGER.log_info("Steps ended early, starting next stage")
                        stage = Stage.FINAL

                case Stage.FINAL:
                    if violation_below_cutoff:
                        num_perms = self.permute_remove_corners(max_angle=1, max_torque=np.inf)
                        if num_perms:
                            LOGGER.log_info("All overlaps removed, additional corners removed, repeating stage")
                        else:
                            LOGGER.log_info("All overlaps removed, stopping iterations")
                            break
                    else:
                        LOGGER.log_warning("Finalizing iteration has not removed all overlaps, aborting")
                        break

            self.permute_update_network()
            self.save_current_network()

            self.step_log_list.append(SimStep())
            if draw_stats:
                self.draw_stats(show=False)
            if draw_vid:
                self.drawer.frame_network_vid(network=self.network, text="Iter {0} permuted".format(iter + 1), show_frame=show_vid_freq is not None)
            if checkpoint_path:
                text = "Iter {0:2d} Mod Complete".format(iter + 1)
                name = "iter_{0:02}_b_post_mod".format(iter + 1)
                self.sol.drawer.save_network_img(source=self.network, text=text, filename=checkpoint_path+name+".png", draw_env=True, draw_forces=True)
                LOGGER.to_json(self.network.to_dict(), path=checkpoint_path, name=name)

        self.draw_stats(show=show_stats)
        if draw_vid:
            self.drawer.frame_network_vid(network=self.network, text="Placement completed".format(iter + 1, stage.name), show_frame=show_vid_freq is not None)
        LOGGER.log_empty()
        pass

    def get_forces_and_scores(self, f_primarcs: float = 20, f_primprims: float = 20, f_arcdists: float = 20, f_arcsteps: float = 10, t_prims: float = 7.5,
                              perim_outline: float = 0.0, force_ramp: float = 0.1, min_angle: float = 75, straighten_corner: bool = True, min_angle_straighten: float = 135) -> bool:
        self.clear_step()
        self.fill_collision_grid(perim_outline=perim_outline)
        if f_primarcs:
            self.get_forces_primarcs(fmax=f_primarcs, perim_outline=perim_outline, force_ramp=force_ramp)
        if f_primprims:
            self.get_forces_primprims(fmax=f_primprims, perim_outline=perim_outline, force_ramp=force_ramp)
        # if f_arcdists: self.get_forces_arcs_min_dist(fmax=f_arcdists, perim_outline=perim_outline, force_ramp=force_ramp) # currently not in use
        if f_arcsteps:
            self.get_forces_arcs_step(fmax=f_arcsteps)
        if t_prims:
            self.get_torques_prims(tmax=t_prims, min_angle=min_angle, straighten_corner=straighten_corner, min_angle_straighten=min_angle_straighten)
        self.get_forces_torque()
        self.get_scores()
        self.safe_forces_comp()

    def log_scores(self, violation_cutoff: float):
        overlap_primarcs = list(sorted(self.overlap_primarcs, key=lambda t: t[1], reverse=True))
        overlap_primprims = list(sorted(self.overlap_primprims, key=lambda t: t[1], reverse=True))
        overlap_arclen = list(sorted(self.overlap_arclen, key=lambda t: t[1], reverse=True))
        violation_primangle = list(sorted(self.violation_primangle, key=lambda t: t[3], reverse=True))

        out = "Highest Violation Score {0:7.5f} above cutoff {1:6.5f}".format(self.max_violation, violation_cutoff)
        if overlap_primarcs:
            out += "\n      Primarc overlaps remain ({0} nonzero)".format(len(overlap_primarcs))
        for arc, score in overlap_primarcs:
            if score < violation_cutoff:
                break
            out += "\n        {0:7.5f}: {1}".format(score, arc)
        if overlap_primprims:
            out += "\n      Primprim overlaps remain ({0} nonzero)".format(len(overlap_primarcs))
        for prim1, score, prim2 in overlap_primprims:
            if score < violation_cutoff:
                break
            out += "\n        {0:7.5f}: {1} x {2}".format(score, prim1, prim2)
        if overlap_arclen:
            out += "\n      Arc length overlaps remain ({0} nonzero)".format(len(overlap_arclen))
        for arc, score in overlap_arclen:
            if score < violation_cutoff:
                break
            out += "\n        {0:7.5f}: {1}".format(score, arc)
        if violation_primangle:
            out += "\n      Prim angle violations remain ({0} nonzero)".format(len(violation_primangle))
        for prim, arc1, arc2, score in violation_primangle:
            if score < violation_cutoff:
                break
            out += "\n        {0:7.5f}: {1}: {2} x {3}".format(score, prim, arc1, arc2)

        LOGGER.log_info(out)

    def step(self, k_step: float, step: int, vel_cutoff: float, k_frict: float = 0.8, vmax: float = 25, verbose: int | None = None) -> bool:
        sum_force = 0
        sum_vel = 0
        sum_move = 0
        max_move = 0
        max_vel_exp = 0
        max_vel_block = 0 if step > 10 else np.nan
        max_pos_block = 0 if step > 25 else np.nan

        self.k_step_vals.append(k_step)
        if step > 10:
            self.k_step_vals.popleft()
        k_step_avg10 = np.average(self.k_step_vals)

        self.max_force = 0
        self.max_vel = 0
        self.max_move_diff = 0 if step > 10 else np.nan

        for prim in self.network.primitives:
            if prim.fixed:
                continue

            oldpos = np.copy(prim.pos)
            force = self.forces[prim] * k_step
            abs_force = np.linalg.norm(force)
            self.max_force = max(abs_force, self.max_force)
            sum_force += abs_force

            self.velocities[prim] = self.velocities[prim] * k_frict + force
            abs_vel = np.linalg.norm(self.velocities[prim])
            if abs_vel > vmax:
                # LOGGER.log_warning("Velocity of {0} large! {1:.1f} > {2}".format(prim, abs_vel, vmax))
                self.velocities[prim] *= vmax / abs_vel

            prim.pos += self.velocities[prim]
            prim.pos[0] = max(prim.size[0] / 2, min(self.env.field_size[0] - prim.size[0] / 2, prim.pos[0]))
            prim.pos[1] = max(prim.size[1] / 2, min(self.env.field_size[1] - prim.size[1] / 2, prim.pos[1]))

            abs_vel = np.linalg.norm(prim.pos - oldpos)
            self.max_vel = max(abs_vel, self.max_vel)
            sum_vel += abs_vel

            abs_move = np.linalg.norm(prim.pos - self.init_pos[prim])
            max_move = max(abs_move, max_move)
            sum_move += abs_move

            if self.vel_exp[prim] is None:
                self.vel_exp[prim] = self.velocities[prim].copy()
            else:
                self.vel_exp[prim] = self.vel_exp[prim] * 0.95 + self.velocities[prim] * (1 - 0.95)
            max_vel_exp = max(max_vel_exp, np.linalg.norm(self.vel_exp[prim]))

            if step > 10:
                max_vel_block = max(max_vel_block, np.linalg.norm(self.vel_block[prim]) / 10)
                old_vel = self.vel_block_vals[prim].popleft()
                self.vel_block[prim] -= old_vel
            self.vel_block_vals[prim].append(self.velocities[prim].copy())
            self.vel_block[prim] += self.velocities[prim]

            if step > 25:
                dists = np.linalg.norm(self.pos_block_vals[prim] - prim.pos, axis=1)
                max_pos_block = max(max_pos_block, np.min(dists))
                self.pos_block_vals[prim].popleft()
            self.pos_block_vals[prim].append(prim.pos.copy())

            if step > 10:
                prim_max_move_diff = (max(self.max_move_vals[prim]) - min(self.max_move_vals[prim])) / 10 / k_step_avg10
                self.max_move_diff = max(self.max_move_diff, prim_max_move_diff)
                self.max_move_vals[prim].popleft()
            self.max_move_vals[prim].append(max_move)

        avg_vel = sum_vel / self.n_movable
        avg_force = sum_force / self.n_movable
        avg_move = sum_move / self.n_movable

        self.step_log.k_step = k_step
        self.step_log.avg_move = avg_move
        self.step_log.max_move = max_move
        self.step_log.avg_vel = avg_vel
        self.step_log.max_vel = self.max_vel
        self.step_log.avg_force = avg_force
        self.step_log.max_force = self.max_force
        self.step_log.max_vel_exp = max_vel_exp
        self.step_log.max_vel_block = max_vel_block
        self.step_log.max_pos_block = max_pos_block
        self.step_log.max_move_diff = self.max_move_diff

    def get_scores(self, overlap_cutoff=0.01):
        # Sum of magnitude of resulting primarc_arc forces of both nodes
        self.score_primarc = {a: 0.0 for a in self.network.arcs}
        for arc in self.network.arcs:
            force_sum = np.zeros(2)
            for name, force in self.forces_comp[arc.tail] + self.forces_comp[arc.head]:
                if name == "primarc_arc":
                    force_sum += force
            score = np.linalg.norm(force_sum)
            self.score_primarc[arc] = score
        self.step_log.score_primarc = max(self.score_primarc.values())
        self.score_primarc = {a: s for a, s in self.score_primarc.items() if s > 0}

        # Sum of individual magnitudes of primarc_arc forces of both nodes
        self.score_primarc_abs = {a: 0.0 for a in self.network.arcs}
        for arc in self.network.arcs:
            score = 0.0
            for name, force in self.forces_comp[arc.tail] + self.forces_comp[arc.head]:
                if name == "primarc_arc":
                    score += np.linalg.norm(force)
            self.score_primarc_abs[arc] = score
        self.step_log.score_primarc_abs = max(self.score_primarc_abs.values())
        self.score_primarc_abs = {a: s for a, s in self.score_primarc_abs.items() if s > 0}

        # Sum of overlap ratios that each are larger than cutoff
        self.score_overlap_cutoff = {a: 0.0 for a in self.network.arcs}
        for arc, score in self.overlap_primarcs:
            if score > overlap_cutoff:
                self.score_overlap_cutoff[arc] += score
        for prim, score, _ in self.overlap_primprims:
            if score > overlap_cutoff:
                for arc in prim.arcs:
                    self.score_overlap_cutoff[arc] += score
        for arc, score in self.overlap_arclen:
            if score > overlap_cutoff:
                self.score_overlap_cutoff[arc] += score
        self.step_log.score_overlap_cutoff = max(self.score_overlap_cutoff.values()) * 50
        self.score_overlap_cutoff = {a: s for a, s in self.score_overlap_cutoff.items() if s > 0}

        # Max of difference of straighten angle violations
        score_straighten_newarcs = {a: 0.0 for a in self.network.arcs}
        for _, arc1, arc2, score in self.violation_primangle_straighten:
            score_straighten_newarcs[arc1] += score
            score_straighten_newarcs[arc2] += score

        if self.score_straighten_oldarcs is None:
            self.score_straighten = np.inf
        else:
            self.score_straighten = max(abs(self.score_straighten_oldarcs[a] - new_s) for a, new_s in score_straighten_newarcs.items() if a in self.score_straighten_oldarcs)
        self.score_straighten_oldarcs = score_straighten_newarcs
        self.step_log.score_straighten = self.score_straighten * 100

        # Sum of all overlap ratios
        self.score_overlap = {a: 0.0 for a in self.network.arcs}
        for arc, score in self.overlap_primarcs:
            self.score_overlap[arc] += score
        for prim, score, _ in self.overlap_primprims:
            for arc in prim.arcs:
                self.score_overlap[arc] += score
        for arc, score in self.overlap_arclen:
            self.score_overlap[arc] += score
        self.step_log.score_overlap = max(self.score_overlap.values()) * 50

        # Sum of all overlap ratios and angle violations
        self.score_violation = self.score_overlap.copy()
        for _, arc1, arc2, score in self.violation_primangle:
            self.score_violation[arc1] += score
            self.score_violation[arc2] += score
        self.step_log.score_violation = max(self.score_violation.values()) * 50

        self.score_overlap = {a: s for a, s in self.score_overlap.items() if s > 0}
        self.score_violation = {a: s for a, s in self.score_violation.items() if s > 0}
        self.step_log.num_violation = len(self.score_violation)

        self.max_violation = max(self.score_violation.values()) if len(self.score_violation) > 0 else 0

    def permute_remove_corners(self, min_length: float = 0.5, max_angle: float = 10, max_torque: float = 5, torque_override_angle: float = 1) -> int:
        def remove_corner_bool(corner: Corner) -> bool:
            arc_angle = corner.get_angles()[1][1]
            torque0 = self.torques_abs[corner.incoming[0]]
            torque1 = self.torques_abs[corner.outgoing[0]]
            r = (corner.incoming[0].length < corner.incoming[0].min_step_length * min_length
                 or corner.outgoing[0].length < corner.outgoing[0].min_step_length * min_length
                 or (arc_angle >= 180 - max_angle and arc_angle <= 180 + max_angle))\
                and (torque0 < max_torque and torque1 < max_torque or arc_angle <= torque_override_angle)
            return r

        corners = list(filter(lambda p: type(p) == Corner and remove_corner_bool(p), self.network.primitives))
        num_perms = len(corners)

        if num_perms == 0:
            LOGGER.log_info("Removing Corners: No corner above threshold, skipping")
            return 0

        out = "Removing Corners: {0}".format(num_perms)
        for corner in corners:
            arc_angle = corner.get_angles()[1][1]
            force = self.forces_abs[corner]
            torque0 = self.torques_abs[corner.incoming[0]]
            torque1 = self.torques_abs[corner.outgoing[0]]
            out += "\n      {0}, angle {1:6.2f}, lengths {2:3.0f}, {3:3.0f}, forces {4:5.2f}, torques {5:5.2f}, {6:5.2f}"\
                .format(corner, arc_angle, corner.incoming[0].length, corner.outgoing[0].length, force, torque0, torque1)
            self.network.mod_remove_corner(corner=corner)
        LOGGER.log_info(out)
        return num_perms

    def permute_split_corners(self, similar_threshold: float = 0.999, primary_threshold: float = 0.025, secondary_threshold: float = 1e-4) -> int:
        def split_corner_bool(arc: Arc) -> bool:
            head_at_border = all(p-s/2 <= 0 or p+s/2 >= f for p, s, f in zip(arc.head.pos, arc.mover_size, self.env.field_size))
            tail_at_border = all(p-s/2 <= 0 or p+s/2 >= f for p, s, f in zip(arc.tail.pos, arc.mover_size, self.env.field_size))
            r = arc.length > arc.min_step_length * 1.25\
                and not (arc.fixed and head_at_border and tail_at_border)
            return r
        primary = self.score_overlap
        secondary = self.score_violation

        out = "Splitting Corners:\n"
        arc_score = []
        for i, (score, threshold) in enumerate(((primary, primary_threshold), (secondary, secondary_threshold))):
            arc_score = list((a, s) for a, s in score.items() if a in self.network.arcs and s > threshold and split_corner_bool(a))
            arc_score.sort(key=lambda a: a[1], reverse=True)
            if len(arc_score) > 0:
                out += "      Using Score {0}\n".format(i)
                break

            arc_score_nonzero = list((a, s) for a, s in score.items() if a in self.network.arcs and s > 1e-4)
            out += "      Score {0} empty: {1} nonzero below threshold\n".format(i, len(arc_score_nonzero))
            # for arc, score in arc_score_nonzero:
            #     out += "        {0:7.5f}: {1}\n".format(score, arc)

        num_perms = len(arc_score)

        if num_perms == 0:
            out += "      No score above threshold, skipping"
            LOGGER.log_info(out)
            return 0

        max_score = max(s for _, s in arc_score)
        to_split = tuple(filter(lambda a: a[1] > similar_threshold * max_score, arc_score))
        to_skip = tuple(filter(lambda a: a[1] <= similar_threshold * max_score, arc_score))

        out += "      Splitting {0}".format(len(to_split))
        for arc, score in to_split:
            out += "\n        {0:7.5f}: {1}".format(score, arc)
            corner = self.network.mod_add_corner(arc=arc)
        out += "\n      {0} below threshold".format(len(to_skip))
        for arc, score in to_skip:
            out += "\n        {0:7.5f}: {1}".format(score, arc)
        LOGGER.log_info(out)
        return num_perms

    def permute_update_network(self):
        self.n_movable = self.network.n_movable
        self.network.assign_ids()
        self.init_pos = {p: p.pos.copy() for p in self.network.primitives}
        self.get_forces_and_scores(min_angle=89)

    def fill_collision_grid(self, perim_outline: float):
        self.collision_gridlen = np.max(self.env.grid_size) * (1 + perim_outline + 0.1)

        self.collision_griddims = np.ceil(self.env.field_size / self.collision_gridlen).astype(int)
        self.collision_grid = np.empty(self.collision_griddims, dtype=set)

        for p, _ in np.ndenumerate(self.collision_grid):
            self.collision_grid[p] = set()
        for prim in self.network.primitives:
            self.collision_grid[*(prim.pos / self.collision_gridlen).astype(int)].add(prim)

    def get_collision_grid_arc(self, arc: Arc) -> set[Primitive]:
        prims = set()
        vect = arc.vect
        dir_x, dir_y = np.sign(vect).astype(int)
        t_delta_x = abs(self.collision_gridlen / vect[0]) if vect[0] != 0 else np.inf
        t_delta_y = abs(self.collision_gridlen / vect[1]) if vect[1] != 0 else np.inf
        x, y = (arc.tail.pos / self.collision_gridlen).astype(int)

        if dir_x == 0:
            t_next_x = np.inf
        elif dir_x <= 0:
            t_next_x = (x * self.collision_gridlen - arc.tail.pos[0]) / vect[0]
        else:
            t_next_x = ((x + 1) * self.collision_gridlen - arc.tail.pos[0]) / vect[0]

        if dir_y == 0:
            t_next_y = np.inf
        elif dir_y <= 0:
            t_next_y = (y * self.collision_gridlen - arc.tail.pos[1]) / vect[1]
        else:
            t_next_y = ((y + 1) * self.collision_gridlen - arc.tail.pos[1]) / vect[1]

        for idx in ((x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y), (x, y), (x+1, y), (x-1, y+1), (x, y+1), (x+1, y+1)):
            if idx[0] < 0 or idx[1] < 0 or any(idx >= self.collision_griddims):
                continue
            prims |= self.collision_grid[idx]

        while t_next_x <= 1 or t_next_y <= 1:
            if t_next_x < t_next_y:
                x += dir_x
                t_next_x += t_delta_x
                if x + dir_x >= self.collision_griddims[0]:
                    continue
                for idx_y in (y-1, y, y+1):
                    if idx_y < 0 or idx_y >= self.collision_griddims[1]:
                        continue
                    prims |= self.collision_grid[x + dir_x, idx_y]
            else:
                y += dir_y
                t_next_y += t_delta_y
                if y + dir_y >= self.collision_griddims[1]:
                    continue
                for idx_x in (x-1, x, x+1):
                    if idx_x < 0 or idx_x >= self.collision_griddims[0]:
                        continue
                    prims |= self.collision_grid[idx_x, y + dir_y]

        prims.remove(arc.tail)
        prims.remove(arc.head)
        return prims

    def get_collision_grid_prim(self, prim: Primitive) -> set[Primitive]:
        x, y = (prim.pos / self.collision_gridlen).astype(int)
        prims = set()
        for idx in ((x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y), (x, y), (x+1, y), (x-1, y+1), (x, y+1), (x+1, y+1)):
            if idx[0] < 0 or idx[1] < 0 or any(idx >= self.collision_griddims):
                continue
            prims |= self.collision_grid[idx]
        prims.remove(prim)
        return prims

    @staticmethod
    def get_collision_primprims(prim1: Primitive, prim2: Primitive) -> tuple[bool, float, float, np.ndarray[float]]:
        vect = prim2.pos - prim1.pos
        vect_abs = abs(vect)
        dist = np.linalg.norm(vect)
        dir = vect / dist
        d_col = prim1.size  # (prim1.size + prim2.size) / 2 if prim1.size != prim2.size
        c = all(vect_abs < d_col)
        if vect_abs[0] == 0:
            dist_col = d_col[0]
        elif vect_abs[1] / vect_abs[0] > d_col[1] / d_col[0]:
            # vector is steeper -> collision with upper/underside
            dist_col = d_col[1] * dist / vect_abs[1]
        else:
            dist_col = d_col[0] * dist / vect_abs[0]
        return c, dist, dist_col, dir

    @staticmethod
    def get_collision_primarcs(prim: Primitive, arc: Arc) -> tuple[bool, float, float, np.ndarray[float], float]:
        vect = arc.vect
        dir = arc.norm
        r = prim.pos-arc.tail.pos
        fract, dist = np.linalg.solve(np.array([vect, dir]).T, r)
        dist_sign = np.sign(dist)
        dir = dir if dist > 0 else -dir
        dist = abs(dist)

        # TODO this is only correct for square movers!
        dist_col = (arc.width + prim.get_width_rot_rad(arc.rot_rad + np.pi / 2)) / 2 - 0.001
        rot_quad = (int)(arc.rot) % 90
        if rot_quad == 0:
            d_fract = 0
        else:
            d_fract = (arc.mover_size[0] - (arc.mover_size[0] + arc.mover_size[1]) * rot_quad / 90) * dist_sign / arc.length
        c = fract > 0.01 + d_fract and fract < 0.99 + d_fract

        return c, dist, dist_col, dir, fract

    def get_forces_primarcs(self, fmax: float, perim_outline: float, force_ramp: float):
        sum_force = 0
        max_force = 0

        for arc in self.network.arcs:
            for prim in self.get_collision_grid_arc(arc):
                if all(arc.tail.pos == arc.head.pos) or all(prim.pos == arc.head.pos) or all(prim.pos == arc.tail.pos):
                    LOGGER.log_error("PrimArc Primitives overlap! {0} x {1}".format(arc, prim))
                    continue
                if arc.fixed and prim.fixed:
                    continue

                c, dist, dist_col, dir, fract = self.get_collision_primarcs(prim=prim, arc=arc)
                if not c:
                    continue

                dist_min = dist_col * (1 + perim_outline)
                if dist >= dist_min:
                    continue

                if dist < dist_col and not (type(arc.tail) is PrimStation or type(arc.head) is PrimStation):
                    score = (dist_col - dist) / dist_col
                    self.overlap_primarcs.append((arc, score))

                if force_ramp == 0:
                    force = fmax
                elif dist == 0:  # overlap
                    force = fmax * 2
                else:
                    dist_fmax = dist_col * (1 - force_ramp)
                    force = fmax * min(1, (dist_min - dist) / (dist_min - dist_fmax))

                force = -1 * dir * force

                if arc.tail.fixed or arc.head.fixed:
                    # only move one edge, fixed still have force as reference
                    force_tail = force
                    force_head = force
                else:
                    # shifted collsion check can cause fract not in [0, 1] -> clamp to ensure force in [0, max]
                    fract_lim = min(1, max(0, fract))
                    force_tail = force * (1 - fract_lim)
                    force_head = force * fract_lim

                self.forces[arc.tail] += force_tail
                self.forces[arc.head] += force_head
                self.forces[prim] -= force

                self.forces_comp[arc.tail].append(("primarc_arc", force_tail))
                self.forces_comp[arc.head].append(("primarc_arc", force_head))
                self.forces_comp[prim].append(("primarc_prim", - force))

                self.forces_abs[arc.tail] += np.linalg.norm(force_tail)
                self.forces_abs[arc.head] += np.linalg.norm(force_head)
                force_abs = np.linalg.norm(force)
                self.forces_abs[prim] += force_abs
                sum_force += force_abs
                max_force = max(max_force, force_abs)

        self.step_log.avg_force_primarcs = sum_force / self.n_movable
        self.step_log.max_force_primarcs = max_force

    def get_forces_primprims(self, fmax: float, perim_outline: float, force_ramp: float):
        sum_force = 0
        max_force = 0

        for prim1 in self.network.primitives:
            for prim2 in self.get_collision_grid_prim(prim1):
                if prim1.fixed and prim2.fixed:
                    continue

                if all(prim2.pos == prim1.pos):
                    LOGGER.log_error("Overlapping PrimPrim! {0}, {1}".format(prim1, prim2))
                    if not prim1.fixed:
                        prim1.pos += np.array([1, 1])
                    else:
                        prim2.pos += np.array([1, 1])

                c, dist, dist_col, dir = self.get_collision_primprims(prim1, prim2)

                if dist < dist_col and not (prim1.fixed and prim2.fixed):
                    score = (dist_col - dist) / dist_col
                    self.overlap_primprims.append((prim1, score, prim2))
                    self.overlap_primprims.append((prim2, score, prim1))

                dist_min = dist_col * (1 + perim_outline)
                if dist >= dist_min:
                    continue

                if force_ramp == 0:
                    force = fmax
                else:
                    dist_fmax = dist_col * (1 - force_ramp)
                    force = fmax * min(1, (dist_min - dist) / (dist_min - dist_fmax))

                force = -1 * dir * force
                self.forces[prim1] += force
                self.forces[prim2] -= force

                self.forces_comp[prim1].append(("primprim", force))
                self.forces_comp[prim2].append(("primprim", - force))

                force_abs = np.linalg.norm(force)
                self.forces_abs[prim1] += force_abs
                self.forces_abs[prim2] += force_abs
                sum_force += force_abs
                max_force = max(max_force, force_abs)

        self.step_log.avg_force_primprims = sum_force / self.n_movable
        self.step_log.max_force_primprims = max_force

    def get_forces_arcs_min_dist(self, fmax: float, perim_outline: float, force_ramp: float):
        sum_force_repr = 0
        max_force_repr = 0

        for arc in self.network.arcs:
            if arc.fixed:
                continue
            if arc.min_spaces == 0:
                continue
            length = arc.length

            dist_col = arc.min_length
            if length < dist_col:
                score = (dist_col - length) / dist_col
                self.overlap_arclen.append((arc, score))

            dist_min = dist_col * (1 + perim_outline)
            if length >= dist_min:
                continue

            dist_fmax = dist_col * (1 - force_ramp)
            if dist_min == dist_fmax:
                force = fmax
            else:
                force = fmax * min(1, (dist_min - length) / (dist_min - dist_fmax))

            force = -1 * arc.dir * fmax

            self.forces[arc.tail] += force
            self.forces[arc.head] -= force

            self.forces_comp[arc.tail].append(("arcs_repr", force))
            self.forces_comp[arc.head].append(("arcs_repr", - force))

            force_abs = np.linalg.norm(force)
            self.forces_abs[arc.tail] += force_abs
            self.forces_abs[arc.head] += force_abs
            sum_force_repr += force_abs
            max_force_repr = max(max_force_repr, force_abs)

        self.step_log.avg_force_arcs_repr = sum_force_repr / self.n_movable
        self.step_log.max_force_arcs_repr = max_force_repr

    def get_forces_arcs_step(self, fmax: float, buffer_spacing: float = 0.01):
        sum_force_attr = 0
        max_force_attr = 0

        for arc in self.network.arcs:
            if arc.fixed:
                continue
            length = arc.length

            step_length = arc.min_step_length * (1 + buffer_spacing)
            steps_target = round(length / step_length)
            dist_target = steps_target * step_length

            if steps_target <= arc.min_spaces:
                continue

            step_err = (length - dist_target) / (step_length / 2)
            if abs(step_err) < buffer_spacing:
                continue

            if abs(step_err) > 0.5:
                step_err = (1 - abs(step_err)) * np.sign(step_err)
            force = fmax * step_err

            force = arc.dir * force

            self.forces[arc.tail] += force
            self.forces[arc.head] -= force

            self.forces_comp[arc.tail].append(("arcs_step", force))
            self.forces_comp[arc.head].append(("arcs_step", - force))

            force_abs = np.linalg.norm(force)
            self.forces_abs[arc.tail] += force_abs
            self.forces_abs[arc.head] += force_abs
            sum_force_attr += force_abs
            max_force_attr = max(max_force_attr, force_abs)

        self.step_log.avg_force_arcs_step = sum_force_attr / self.n_movable
        self.step_log.max_force_arcs_step = max_force_attr

    def get_torques_prims(self, tmax: float, min_angle: float, straighten_corner: bool, min_angle_straighten: float):
        sum_torque = 0
        max_torque = 0

        for prim in self.network.primitives:
            match prim:
                case Iface() | Merge() | Split() | Cross() | Corner():
                    name = {Iface: "iface", Merge: "merge", Split: "split", Cross: "cross", Corner: "corner"}[type(prim)]
                    arcs_angles = prim.get_angles()
                    torques_comp = {a: 0.0 for a in prim.arcs}
                    for (arc1, rot1), (arc2, rot2) in zip(arcs_angles[:-1], arcs_angles[1:]):
                        drot = rot2 - rot1
                        if straighten_corner and type(prim) in (Corner, Iface):
                            if (drot >= 90 and drot < min_angle_straighten) or drot >= 179:
                                continue
                        else:
                            if drot >= 90:
                                continue

                        if straighten_corner and type(prim) in (Corner, Iface) and drot < min_angle_straighten:
                            score = (min_angle_straighten - drot) / min_angle_straighten
                            self.violation_primangle_straighten.append((prim, arc1, arc2, score))

                        if drot < min_angle:
                            score = (min_angle - drot) / min_angle
                            self.violation_primangle.append((prim, arc1, arc2, score))

                        if straighten_corner and type(prim) in (Corner, Iface) and drot >= min_angle_straighten:
                            if drot < min_angle_straighten + 90 - min_angle and min_angle < 90:
                                # ramp at min_angle_straighten
                                torque = tmax * min(0.25, (drot - min_angle_straighten) / (90 - min_angle))
                            else:
                                torque = tmax * min(0.25, (180 - drot) / (180 - min_angle_straighten))
                        elif min_angle == 90:
                            torque = tmax
                        else:
                            torque = tmax * min(1, (90 - drot) / (90 - min_angle))

                        self.torques[arc1] -= torque
                        self.torques[arc2] += torque

                        torques_comp[arc1] -= torque
                        torques_comp[arc2] += torque

                        torque_abs = abs(torque)
                        self.torques_abs[arc1] += torque_abs
                        self.torques_abs[arc2] += torque_abs
                        sum_torque += torque_abs
                        max_torque = max(max_torque, torque_abs)

                    for arc, torque in torques_comp.items():
                        if torque != 0.0:
                            self.torques_comp[arc].append((name, torque))

                case _: continue

        self.step_log.avg_torque = sum_torque / self.n_movable
        self.step_log.max_torque = max_torque

    def get_forces_torque(self):
        sum_torque = 0
        max_torque = 0

        for arc in self.network.arcs:
            for name, torque in self.torques_comp[arc]:
                force = arc.norm * torque
                self.forces[arc.tail] += force
                self.forces[arc.head] -= force
                torque_abs = abs(torque)
                sum_torque += torque_abs
                max_torque = max(max_torque, torque_abs)

                self.forces_comp[arc.tail].append(("f_torque:" + name, force))
                self.forces_comp[arc.head].append(("f_torque:" + name, - force))

                force_abs = np.linalg.norm(force)
                self.forces_abs[arc.tail] += force_abs
                self.forces_abs[arc.head] += force_abs

        self.step_log.avg_force_torque = sum_torque / self.n_movable
        self.step_log.max_force_torque = max_torque

    def safe_forces_comp(self):
        self.network.forces_comp = self.forces_comp
        self.network.torques_comp = self.torques_comp

    def save_current_network(self):
        self.sol.networks.insert(len(self.sol.networks) - 1, self.network.copy())

    def clear_iter(self):
        self.velocities = {p: np.zeros(2, float) for p in self.network.primitives}
        self.vel_exp = {p: None for p in self.network.primitives}
        self.vel_block = {p: np.zeros(2, float) for p in self.network.primitives}
        self.vel_block_vals = {p: deque() for p in self.network.primitives}
        self.pos_block_vals = {p: deque() for p in self.network.primitives}
        self.max_move_vals = {p: deque() for p in self.network.primitives}
        self.k_step_vals = deque()
        self.min_max_force = np.inf

    def clear_step(self):
        self.forces = {p: np.zeros(2, dtype=float) for p in self.network.primitives}
        self.forces_abs = {p: 0.0 for p in self.network.primitives}
        self.forces_comp = {p: [] for p in self.network.primitives}
        self.torques = {a: 0.0 for a in self.network.arcs}
        self.torques_abs = {a: 0.0 for a in self.network.arcs}
        self.torques_comp = {a: [] for a in self.network.arcs}
        self.overlap_primarcs = []
        self.overlap_primprims = []
        self.overlap_arclen = []
        self.violation_primangle = []
        self.violation_primangle_straighten = []
        self.score_directed_force = None
        self.score_primarc_abs = None
        self.score_primarc_abs = None
        self.score_overlap_cutoff = None
        self.score_overlap = None
        self.score_violation = None

    def draw_stats(self, show: bool = True, save: bool = True, filename: str = './output/force_placment.png') -> plt.Figure:
        num_points = len(self.step_log_list)
        data: dict[str, np.ndarray[int | float]] = {}
        for p, _ in vars(SimStep()).items():
            data[p] = np.empty(num_points)
        for i, entry in enumerate(self.step_log_list):
            for p, val in vars(entry).items():
                data[p][i] = val

        f, axs = plt.subplots(9, 1, sharex=True, figsize=(25, 12))
        f.set_size_inches((25.5, 12.5))
        f.subplots_adjust(hspace=0.05)
        axs[-1].set_xlabel("Steps per Iteration")

        ax = axs[0]
        ax.set_ylabel('Step Gain')
        ax.plot(data["k_step"])

        ax = axs[1]
        ax.set_ylabel('Violations')
        ax.plot(data["num_violation"])

        ax = axs[2]
        ax.set_ylabel('Moves')
        ax.plot(data["avg_move"])
        ax.plot(data["max_move"])
        ax.legend(["Average", "Max"], loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[3]
        ax.set_ylabel('Velocities')
        ax.plot(data["avg_vel"])
        ax.plot(data["max_vel"])
        ax.plot(data["max_vel_exp"])
        ax.plot(data["max_vel_block"])
        ax.plot(data["max_pos_block"])
        ax.legend(["Average", "Max", "Max Vel EMA", "Max Vel Block", "Max Pos Block"], loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[4]
        ax.set_ylabel('Forces')
        ax.plot(data["avg_force"])
        ax.plot(data["max_force"])
        ax.legend(["Average", "Max"], loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[5]
        ax.set_ylabel('Comp Avg')
        ax.plot(data["avg_force_primarcs"])
        ax.plot(data["avg_force_primprims"])
        ax.plot(data["avg_force_arcs_repr"])
        ax.plot(data["avg_force_arcs_step"])
        ax.plot(data["avg_torque"])
        ax.plot(data["avg_force_torque"])
        ax.legend(["PrimArcs", "PrimPrims", "Arcs Repr", "Arcs Attr", "Force_Torque", "Torque"], loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[6]
        ax.set_ylabel('Comp Max')
        ax.plot(data["max_force_primarcs"])
        ax.plot(data["max_force_primprims"])
        ax.plot(data["max_force_arcs_repr"])
        ax.plot(data["max_force_arcs_step"])
        ax.plot(data["max_force_torque"])
        ax.plot(data["max_torque"])
        ax.legend(["PrimArcs", "PrimPrims", "Arcs Repr", "Arcs Attr", "Force_Torque", "Torque"], loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[7]
        ax.set_ylabel('Scores')
        ax.plot(data["score_primarc"])
        ax.plot(data["score_primarc_abs"])
        ax.plot(data["score_overlap"])
        ax.plot(data["score_overlap_cutoff"])
        ax.plot(data["max_move_diff"])
        ax.legend(["Primarc", "Primarc Abs", "Overlap", "Overlap Cutoff", "Move Diff"], loc='center left', bbox_to_anchor=(1, 0.5))

        ax = axs[8]
        ax.set_ylabel('Scores 2')
        ax.plot(data["max_move_diff"])
        ax.legend(["Move Diff"], loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_yscale('log')

        splits = np.hstack([0, (data["step"] == 0).nonzero()[0][1:], num_points])
        ticks = []
        labels = []
        for prev, next in zip(splits[:-1], splits[1:]):
            center = (int)((prev + next) / 2) - 2
            end = next - 2
            ticks.append(center)
            ticks.append(end)
            stage = data["stage"][end]
            labels.append(Stage(stage).name if not np.isnan(stage) else "")
            labels.append(data["step"][end].astype(int) + 1)

        axs[-1].set_xticks(ticks, labels)
        # axs[-1].set_xticks(splits - 1, data["step"][splits - 2].astype(int) + 1)

        for ax in axs[:-1]:
            ax.set_xlim([0, num_points])
            ax.set_ylim([0, ax.get_ylim()[1] * 1.05])
            ax.vlines(splits - 1, 0, ax.get_ylim()[1], colors=("grey", 0.5))

        for ax in axs[-1:]:
            ax.set_xlim([0, num_points])
            ax.set_ylim([ax.get_ylim()[0] * 0.95, ax.get_ylim()[1] * 1.05])
            ax.vlines(splits - 1, 0, ax.get_ylim()[1], colors=("grey", 0.5))

        if save:
            plt.savefig(filename)
            if not show:
                plt.close()
        if show:
            plt.show(block=True)
