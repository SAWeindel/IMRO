import numpy as np
from bisect import *

from definition import *
from tools.log import LOGGER

from online.controller import Mover, Controller


class Simulator:
    def __init__(self, sce: Scenario, sol: Solution, seq: Sequence = None):
        self.sce = sce
        self.sol = sol

        self.seq = seq or Sequence(sol=self.sol, sce=self.sce)
        self.movers = [Mover(id=i) for i in range(self.sce.mover_num)]

        self.planning_order: list[Mover] = []
        self.time: float = None

        self.iface = StationInterface(sce=self.sce, seq=self.seq, use_rand=True)
        self.controller = Controller(sol=sol)

        self.time_startup_end: float = None
        self.delivery_counter: dict[Station, int] = None
        self.delivery_rate: float = None

    def get_next_mover(self) -> Mover:
        movers = (m for m in self.movers if m.state not in (MoverState.BL_PATH, MoverState.BL_ITEM))
        movers = sorted(movers, key=lambda m: m.next_time_to_plan)
        if len(movers) == 0:
            return None
        return sorted(movers, key=lambda m: m.next_time_to_plan)[0]

    def interact_with_station(self, mover: Mover, start_time: float):
        station = mover.target_node.station
        mover.set_state(MoverState.INTERACTING, time=start_time)

        # happens when init is on path to or at receiver
        if mover.item is None and station.is_receiver:
            mover.items.append((start_time, Item.empty()))
            return

        # startup ends if first mover has reached receiver
        if station.is_receiver:
            if self.time_startup_end is None:
                self.time_startup_end = self.time
            self.delivery_counter[station] += 1

        event = self.iface.interact(station=station, mover=mover, time=start_time)
        mover.next_time_to_plan = event.time_completes
        if event.is_loading:
            if mover.item and mover.item != 0:
                LOGGER.log_warning("Non-Empty mover {0} interacted with Provider: {1}".format(mover, event))
            mover.items.append((event.time_completes, event.item))
        else:
            if mover.item != event.item:
                LOGGER.log_warning("Wrong item delivered to Receiver by mover {0}: Received {1} expected {2}, {3}".format(mover, mover.item, event.item, event))
            mover.items.append((event.time_completes, Item.empty()))

    def run(self):
        self.controller.setup(self.iface, self.movers)

        self.time = 0.0
        self.delivery_counter = {s: 0 for s in self.sol.env.receivers}

        while self.time < self.sce.time_max:
            if all(m.state in (MoverState.BL_PATH, MoverState.BL_ITEM, MoverState.BL_NODE) for m in self.movers):
                time = max(m.actions[-1].time_goal for m in self.movers if m.actions) if any(m.actions for m in self.movers) else 1
                LOGGER.log_error("Deadlock Reached at time {0:6.2f}".format(time))
                self.sce.time_max = np.ceil(time)
                break

            mover = self.get_next_mover()
            self.time = mover.next_time_to_plan
            LOGGER.log_print_line("simulation", "{0:6.2f} Next sim step for Mover {1}, {2}".format(self.time, mover, mover.state.name))

            match mover.state:
                case MoverState.TO_STATION:
                    self.interact_with_station(mover=mover, start_time=self.time)
                    if hasattr(mover, 'current_section'):
                        # tracking of completed paths
                        if mover.last_station is not None:
                            mover.paths.append(self.sol.graph.paths[(mover.last_station, mover.target_node)])
                        mover.last_station = mover.target_node

                        mover.current_section = None
                case MoverState.TO_BL_TIME:
                    mover.set_state(MoverState.BL_TIME, time=self.time)
                case MoverState.TO_BL_PATH:
                    mover.set_state(MoverState.BL_PATH, time=self.time)
                case MoverState.TO_BL_ITEM:
                    mover.set_state(MoverState.BL_ITEM, time=self.time)
                case MoverState.TO_BL_NODE:
                    mover.set_state(MoverState.BL_NODE, time=self.time)
                case _:
                    self.controller.get_actions(mover=mover, time_planning=self.time)

            mover_deps: dict[Mover, list[int]] = {m: [] for m in self.movers}
            if hasattr(self.controller, 'mover_dependencies'):
                for blocking, list_blocked in self.controller.mover_dependencies.items():
                    for blocked in list_blocked:
                        mover_deps[blocked].append(blocking.id)

            item_deps: dict[Mover, list[str]] = {m: [] for m in self.movers}
            if hasattr(self.controller, 'item_dependencies'):
                for blocking, list_blocked in self.controller.item_dependencies.items():
                    for blocked in list_blocked:
                        item_deps[blocked].append(str(blocking))

            out = "{0:6.2f} Sim step complete:\n\t".format(self.time)
            for mover in self.movers:
                out += "{0}: ({1}, {2:.2f}, {3}, {4}), "\
                    .format(mover, mover.state.name, mover.next_time_to_plan, mover_deps[mover], item_deps[mover])
            LOGGER.log_print_line("simulation", out[:-2] + "\n")

        if self.time_startup_end:
            deliveries = np.array([i for i in self.delivery_counter.values()])
            sum_deliveries = np.sum(deliveries).astype(int)
            sum_flows = np.sum([s.flow for s in self.delivery_counter.keys()])
            self.delivery_rate = sum_deliveries / (self.time - self.time_startup_end) * 60
            expected_rate = sum_flows * 60
            out = "Simulation complete after {0:6.2f}s, startup done after {1:6.2f}s:\n".format(self.time, self.time_startup_end)
            out += "      {0} items delivered, {1:5.1f} / {2:5.1f} 1/min ({3:4.0%})\n"\
                .format(sum_deliveries, self.delivery_rate, expected_rate, self.delivery_rate / expected_rate)
            for station, num in self.delivery_counter.items():
                out += "      Station {0:60s}{1:3d} items, {2:5.1f} / {3:5.1f} 1/min ({4:4.0%})\n"\
                    .format(str(station), num, num / (self.time - self.time_startup_end) * 60, station.flow * 60, num / (self.time - self.time_startup_end) / station.flow)
            LOGGER.log_info(out)
        self.seq.movers = self.movers
        return self.seq

    def export_sequence(self, video_filename: str | None = './output/controller_sequence.avi', pickle_filename: str | None = './output/seq.p') -> Sequence:
        LOGGER.log_seq(self.seq)
        if video_filename:
            self.seq.drawer.save_sequence_vid(filename=video_filename)
        if pickle_filename:
            self.seq.to_pickle(filename=pickle_filename)

    @classmethod
    def from_sequence(cls, seq: Sequence):
        return cls(sce=seq.sce, sol=seq.sol, seq=seq)
