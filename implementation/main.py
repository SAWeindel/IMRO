from __future__ import annotations
from online.simulator import Simulator
from offline.planner import Planner
from tools.gui import GUI
from tools.log import LOGGER
from definition import *
from ctypes import *
import os
import shutil
from datetime import datetime


def main():
    # (0: gen sol and seq, 1: gen sol, 2: load and show sol, 3: load sol, gen seq, 4: load and show seq, 5: series of runs, 6: range of simulated mover nums)
    mode = 0

    # Name of the file in ./tasks containing the environment specificaitons
    env_name = "1_1"

    # Specifies a folder in ./runs, which will be used as reference for the run. If None, any initial files will be taken from ./output.
    run_folder = None
    # run_folder = './runs/' + '___' + '/'

    plan_init_flow = None
    # plan_init_flow = 'flow.json'

    plan_init_network = None
    # plan_init_network = 'iter_00.json'

    use_alltoall = False
    split_ifaces = False
    use_merge_split = False
    max_length = 5
    reverse = False

    time_max = 60.0
    mover_num = 20
    # For mode 6
    mover_range_min = 10
    mover_range_max = 30

    save_run = True
    unique_run = True
    gen_seq_vid = True
    save_vid = True

    if mode == 5:
        LOGGER.verbose = 0
        for env_name, split_ifaces, use_merge_split, mover_num in (
            ('1_1', False, False, 24), ('1_1', False, True, 20), ('1_2', False, True, 20), ('1_4', False, True, 16),
            ('2_2', True, False, 30), ('2_2', True, True, 30),
            ('3_1', True, False, 25), ('3_1', True, True, 25), ('3_2', True, False, 25), ('3_2', True, True, 25),
            ('4_1', True, False, 25), ('4_1', True, True, 25), ('4_3', True, True, 25),
            ('5_1', True, False, 20), ('5_1', True, True, 20), ('5_2', True, False, 30), ('5_2', True, True, 30),
        ):
            LOGGER.reset()
            print(env_name, split_ifaces, use_merge_split, max_length, mover_num)
            Run(env_name=env_name, gen_sol=True, gen_seq=True, split_ifaces=split_ifaces, use_merge_split=use_merge_split, max_length=max_length, mover_num=mover_num, time_max=time_max, save_run=True, unique_run=True, save_vid=True)
    elif mode == 6:
        LOGGER.verbose = 0
        Run(env_name=env_name, run_folder=run_folder, gen_sol=True, gen_seq=False, show_sol=False, show_seq=False,
            plan_init_flow=plan_init_flow, plan_init_network=plan_init_network, use_alltoall=use_alltoall, split_ifaces=split_ifaces, use_merge_split=use_merge_split, max_length=max_length,
            mover_num=None, time_max=None, save_run=save_run, unique_run=unique_run, save_vid=save_vid, gen_seq_vid=gen_seq_vid)
        for i in range(mover_range_min, mover_range_max+1):
            LOGGER.reset()
            run = Run(env_name=env_name, run_folder=run_folder, gen_sol=False, gen_seq=True, show_sol=False, show_seq=False,
                      plan_init_flow=plan_init_flow, plan_init_network=plan_init_network, use_alltoall=use_alltoall, split_ifaces=split_ifaces, use_merge_split=use_merge_split, max_length=max_length,
                      mover_num=i, time_max=time_max, save_run=save_run, unique_run=unique_run, save_vid=save_vid, gen_seq_vid=gen_seq_vid)
            print("{0:3d}: {1},".format(i, "{0:5.1f}".format(run.simulator.delivery_rate) if run.simulator.delivery_rate else "DNF"))
    else:
        Run(env_name=env_name, run_folder=run_folder, gen_sol=mode in (0, 1), gen_seq=mode in (0, 3), show_sol=mode in (1, 2), show_seq=mode in (0, 3, 4),
            plan_init_flow=plan_init_flow, plan_init_network=plan_init_network, use_alltoall=use_alltoall, split_ifaces=split_ifaces, use_merge_split=use_merge_split, max_length=max_length, reverse=reverse,
            mover_num=mover_num, time_max=time_max, save_run=save_run, unique_run=unique_run, save_vid=save_vid, gen_seq_vid=gen_seq_vid)


class Run:
    def __init__(self, env_name: str, run_folder: str = None, plan_init_flow: str = None, plan_init_network: str = None, save_run: bool = True, unique_run: bool = True, save_vid: bool = True,
                 gen_sol: bool = False, show_sol: bool = False, gen_seq: bool = False, gen_seq_vid: bool = False, show_seq: bool = False,
                 use_alltoall: bool = None, split_ifaces: bool = None, use_merge_split: bool = None, max_length: int = None, reverse: bool = None,
                 mover_num: int = None, time_max: float = None):
        self.env_name = env_name

        self.save_run = save_run and (gen_sol or gen_seq)
        self.save_vid = save_vid
        self.start_time = datetime.now()

        if run_folder:
            self.in_run = True
            self.env_file = run_folder + 'env_{0}.json'.format(env_name)
            self.seq_file = run_folder + 'seq.p'
            self.sol_file = run_folder + 'sol.p'
            self.plan_init_flow = run_folder + plan_init_flow if plan_init_flow else None
            self.plan_init_network = run_folder + plan_init_network if plan_init_network else None
            self.run_folder = run_folder
        else:
            self.in_run = False
            self.env_file = './tasks/env_{0}.json'.format(env_name)
            self.seq_file = './output/seq.p'
            self.sol_file = './output/sol.p'
            self.plan_init_flow = './output/' + plan_init_flow if plan_init_flow else None
            self.plan_init_network = './output/' + plan_init_network if plan_init_network else None
            self.run_folder = './runs/run_env{0}{1}_{2}{3}{4}/'\
                .format(env_name, 'sm' if use_merge_split else '', 'sol' if gen_sol else '', 'seq' if gen_seq else '',
                        '_' + self.start_time.strftime('%y%m%d-%H%M%S') if unique_run else '')

        self.gen_sol = gen_sol
        self.load_sol = not self.gen_sol and (show_sol or gen_seq)
        self.show_sol = show_sol and not show_seq

        self.gen_seq = gen_seq
        self.load_seq = not self.gen_seq and (show_seq or gen_seq_vid)
        self.gen_seq_vid = gen_seq_vid
        self.show_seq = show_seq

        self.use_alltoall = use_alltoall
        self.split_ifaces = split_ifaces
        self.use_merge_split = use_merge_split
        self.max_length = max_length
        self.reverse = reverse

        self.mover_num = mover_num
        self.time_max = time_max

        self.env: Environment = None
        self.sol: Solution = None
        self.sce: Scenario = None
        self.seq: Sequence = None
        self.gui: GUI = None

        if self.gen_sol:
            self.env = Environment.from_json(self.env_file)
            LOGGER.log_perf_mark("Planner init")
            init_flow = Network.from_json(filename=self.plan_init_flow, env=self.env) if self.plan_init_flow else None
            init_network = Network.from_json(filename=self.plan_init_network, env=self.env) if self.plan_init_network else None
            planner = Planner(env=self.env, init_flow=init_flow, init_network=init_network)
            self.sol = planner.solve(use_alltoall=self.use_alltoall, split_ifaces=self.split_ifaces, use_merge_split=self.use_merge_split, max_length=self.max_length, reverse=self.reverse)
            LOGGER.log_perf_mark("Planner done")
            LOGGER.log_sol(self.sol)
            self.sol.to_pickle(self.sol_file)
        elif self.load_sol:
            self.sol = Solution.from_pickle(self.sol_file)
            self.env = self.sol.env
            LOGGER.sol = self.sol

        if self.show_sol:
            LOGGER.log_num_warning()
            LOGGER.log_perf()
            if self.save_run:
                self.copy_files()
            self.gui = GUI(sol=self.sol)
            LOGGER.log_perf_mark("GUI loaded")
            self.gui.run()
            return

        if self.gen_seq:
            self.sce = Scenario(name="DUMMY", env=self.env, mover_num=self.mover_num, time_max=self.time_max)
            self.simulator = Simulator(sce=self.sce, sol=self.sol)
            LOGGER.log_perf_mark("Simulator Init")
            self.seq = self.simulator.run()
            LOGGER.log_perf_mark("Simulator Done")
            LOGGER.log_seq(self.seq)
            self.seq.to_pickle(self.seq_file)
        elif self.load_seq:
            self.seq = Sequence.from_pickle(self.seq_file)
            self.sol = self.seq.sol
            self.env = self.sol.env
            LOGGER.seq = self.seq
            LOGGER.sol = self.sol

        if self.show_seq:
            if self.gen_seq_vid:
                LOGGER.log_perf_mark("Generating Sequence Video")
                self.seq.drawer.save_sequence_vid()
            LOGGER.log_num_warning()
            LOGGER.log_perf()
            if self.save_run:
                self.copy_files()
            self.gui = GUI(seq=self.seq)
            LOGGER.log_perf_mark("GUI loaded")
            self.gui.run()
        else:
            if self.seq and self.gen_seq_vid:
                LOGGER.log_perf_mark("Generating Sequence Video")
                self.seq.drawer.save_sequence_vid()
            LOGGER.log_perf_mark("Finalized")
            LOGGER.log_num_warning()
            LOGGER.log_perf()
            if self.save_run:
                self.copy_files()

    def copy_files(self):
        timestamp = datetime.timestamp(self.start_time)

        if not self.in_run:
            os.makedirs(self.run_folder, exist_ok=True)

            shutil.copy(src=self.env_file, dst=self.run_folder)

        for f in os.listdir('./output/'):
            srcpath = os.path.join('./output/', f)
            if f.endswith(".avi") or os.path.getmtime(srcpath) < timestamp:
                continue
            shutil.copy(src=srcpath, dst=self.run_folder)

        if self.save_vid:
            vid_file = './output/force_placement.avi'
            if os.path.exists(vid_file) and os.path.getmtime(vid_file) >= timestamp:
                shutil.copy(src=vid_file, dst=self.run_folder)
            vid_file = './output/controller_sequence.avi'
            if os.path.exists(vid_file) and os.path.getmtime(vid_file) >= timestamp:
                shutil.copy(src=vid_file, dst=self.run_folder)


main()
