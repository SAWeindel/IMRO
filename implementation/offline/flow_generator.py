import numpy as np
import itertools
import functools
import operator

from definition import *
from tools.log import LOGGER


class FlowGenerator:
    def __init__(self, env: Environment, sol: Solution):
        self.env = env
        self.sol = sol

        self.stations: np.ndarray[Station] = np.array(self.env.stations)
        self.prim_stations: np.ndarray[PrimStation] = None
        self.num_ifaces: int = None

        self.station_distances: np.ndarray[float] = None                    # Station x Station -> L2 distance (mm)

        self.item_map: dict[Item, int] = None                               # Item -> Item Idx
        self.item_map_inv: dict[int, Item] = None                           # Item Idx -> Item
        self.sources: np.ndarray[float] = None                              # Item Idx x Station Idx -> Total Flow (>0 if provided)

        # self.num_flows: int = None
        self.flows = np.zeros((len(self.stations), len(self.stations)))     # Provider Idx x Receiver Idx -> Flow from P to R (>0 if existing)
        self.num_provided: list[int] = None                                 # Station Idx -> Num Flows Outgoing
        self.num_received: list[int] = None                                 # Station Idx -> Num Flows Incoming
        self.flow_iface_assignment: np.ndarray[int] = None                  # [Assignment] -> [Flow Idx -> [Provider Iface Idx, Receiver Iface Idx]]
        self.station_ifaces: list[list[Iface]] = None                       # Station Idx -> list[Iface]
        self.station_iface_assignment: list[list[int]] = None               # Station Idx -> list[Iface Idx]
        self.center_ifaces: list[int] = []                                  # list[Iface Idx], central interface if station has three interfaces
        self.flows_dict: dict[tuple[Station, Station, Item], float] = {}    # Provider x Receiver x Item -> Flow of Item from P to R (>0)
        self.arc_flow: dict[Arc, float] = {}

        self.arcs: np.ndarray[Arc] = None                                   # Arc Idx -> Arc, connecting interface to interface
        self.opposing_arcs: np.ndarray[int | None] = None                   # Arc Idx -> Arc Idx if Arcs connect same ifaces in opposing flow directions
        self.arc_assignments: np.ndarray[int] = None                        # Iface Idx x Iface Idx -> Arc Idx, connecting provider to receiver
        self.arc_assignments_inv: np.ndarray[int] = None
        self.intersections: np.ndarray[bool] = None                         # Arc Idx x Arc Idx -> bool, true if arcs overlap

        # self.fixed_arcs = None

        self.station_map: np.ndarray = None
        self.num_perm: int = None
        self.num_perm_digits: int = None
        self.min_intersections: int = None
        self.min_ifaces_shared: int = None
        self.min_count: int = None
        self.min_frequency_arr: np.ndarray[float] = None
        self.solution_score_arr: np.ndarray[float] = None
        self.solution_arcs_arr: np.ndarray[bool] = None

        # self.clusters: list[Cluster] = None

        self.solution_ifaces: list[Iface] = None
        self.solution_arcs: list[Arc] = None
        self.solution_primitives: list[Primitive] = None

        self.generate_station_distances()

    def run(self, use_alltoall: bool = False, split_ifaces: bool = False):
        self.generate_sources()
        LOGGER.log_info("Sources: Item x Station\n{0}".format(self.sources))

        if use_alltoall:
            self.generate_flows_alltoall()
        else:
            self.generate_flows()
        LOGGER.log_info("Flows: Provider x Receiver\n{0}".format(self.flows))

        self.generate_iface_assignment()

        self.generate_ifaces()
        # LOGGER.log_info("Iface Assignments: Station\n{0}".format(self.iface_assignment))

        self.generate_arcs()
        # LOGGER.log_info("Edges: Iface x Iface: Edge\n{0}".format(self.edge_assignments))
        # LOGGER.log_info("Intersections: Edge x Edge\n{0}".format(self.intersections))

        self.generate_station_map()

        LOGGER.log_perf_mark("Prepped search")
        self.generate_flowgraph()
        # self.generate_flowgraph(count_abort=10_000)

        if split_ifaces:
            self.generate_station_map_reduced()
            self.generate_flowgraph(count_shared=True)

        LOGGER.log_info("{0:.2f} intersection score, {1} shared ifaces found in {2} ({3}%) permutations"
                        .format(self.min_intersections, self.min_ifaces_shared, self.min_count, round(self.min_count / self.num_perm * 100, 3)))

        self.generate_flow()

    def generate_sources(self):
        self.item_map = {i: idx for idx, i in enumerate(self.env.items)}
        self.item_map_inv = {idx: i for idx, i in enumerate(self.env.items)}

        self.sources = np.zeros((len(self.env.items) + 1, len(self.stations)))  # Item x Station

        for station_idx, station in enumerate(self.stations):
            flow = 1 / station.period / len(station.item_sequence) * (1 if station.is_provider else -1)
            for item in station.item_sequence:
                self.sources[0][station_idx] -= flow
                self.sources[self.item_map[item]][station_idx] += flow

        if any(np.abs(np.sum(self.sources, axis=0)) > 0.0001) or any(np.abs(np.sum(self.sources, axis=1)) > 0.0001):
            raise Exception("Sources not balanced")

    def generate_station_distances(self):
        self.station_distances = np.empty((len(self.stations), len(self.stations)))
        for idx1, station1 in enumerate(self.stations):
            for idx2, station2 in enumerate(self.stations):
                self.station_distances[idx1][idx2] = np.linalg.norm(station1.pos - station2.pos)
                self.station_distances[idx2][idx1] = self.station_distances[idx1][idx2]

    def generate_flows(self):
        sources_rem = np.copy(self.sources)

        for item_idx, station_flows in enumerate(self.sources):
            num_providers = np.sum(1 for f in station_flows if f > 0)
            num_receivers = np.sum(1 for f in station_flows if f < 0)
            if num_providers == 1:
                provider = np.argmax(station_flows)
                receivers = (self.sources[item_idx, :] < 0)
                self.flows[provider, receivers] -= self.sources[item_idx, receivers]
                sources_rem[item_idx, receivers] -= self.sources[item_idx, receivers]
                sources_rem[item_idx, provider] += np.sum(self.sources[item_idx, receivers])
                for receiver in receivers.nonzero()[0]:
                    self.flows_dict[(self.stations[provider], self.stations[receiver], self.item_map_inv[item_idx])] = -self.sources[item_idx, receiver]
                continue
            if num_receivers == 1:
                receiver = np.argmin(station_flows)
                providers = (self.sources[item_idx, :] > 0)
                self.flows[providers, receiver] += self.sources[item_idx, providers]
                sources_rem[item_idx, providers] -= self.sources[item_idx, providers]
                sources_rem[item_idx, receiver] += np.sum(self.sources[item_idx, providers])
                for provider in providers.nonzero()[0]:
                    self.flows_dict[(self.stations[provider], self.stations[receiver], self.item_map_inv[item_idx])] = self.sources[item_idx, provider]
                continue

            for receiver in (sources_rem[item_idx] < 0).nonzero()[0]:
                remainder = sources_rem[item_idx, receiver]
                providers = np.argsort(self.station_distances[receiver])
                for provider in providers[sources_rem[item_idx, providers] > 0]:
                    available = sources_rem[item_idx, provider]
                    if available >= -remainder:
                        self.flows[provider, receiver] -= remainder
                        sources_rem[item_idx, provider] += remainder
                        sources_rem[item_idx, receiver] = 0
                        self.flows_dict[(self.stations[provider], self.stations[receiver], self.item_map_inv[item_idx])] = -remainder
                        break
                    elif available > 0:
                        self.flows[provider, receiver] += available
                        sources_rem[item_idx, provider] = 0
                        remainder += available
                        self.flows_dict[(self.stations[provider], self.stations[receiver], self.item_map_inv[item_idx])] = available

        if np.any(sources_rem):
            raise Exception("Flow could not be assigned")

        # self.num_flows = (self.flows != 0).sum()
        self.num_provided = np.count_nonzero(self.flows, axis=1)
        self.num_received = np.count_nonzero(self.flows, axis=0)

    def generate_flows_alltoall(self):
        for _, station_flows in enumerate(self.sources):
            providers = station_flows > 0
            receivers = station_flows < 0
            for i, f in enumerate(station_flows):
                if providers[i]:
                    self.flows[i, receivers] = f
            # x = station_flows[providers]
            # self.flows[providers, receivers] = station_flows[providers]

        for provider in self.stations:
            for receiver in self.stations:
                for item in self.env.items:
                    self.flows_dict[(provider, receiver, item)] = 1.0

        # self.num_flows = (self.flows != 0).sum()
        self.num_provided = np.count_nonzero(self.flows, axis=1)
        self.num_received = np.count_nonzero(self.flows, axis=0)

    def generate_iface_assignment(self):
        # For each flow, assign a unique flow_iface index to both provider and receiver
        # Iface indices are grouped by corresponding station, provided flows before received
        # This is later required to map the station-internal flow -> iface assignment to the corresponding global flow
        flow_iface_assignment = []
        flow_provider_idx = np.append([0], np.cumsum(np.add(self.num_provided, self.num_received))[:-1])
        flow_receiver_idx = flow_provider_idx + self.num_provided
        for (provider_idx, receiver_idx), flow in np.ndenumerate(self.flows):
            if flow == 0:
                continue
            flow_iface_assignment.append([flow_provider_idx[provider_idx], flow_receiver_idx[receiver_idx]])
            flow_provider_idx[provider_idx] += 1
            flow_receiver_idx[receiver_idx] += 1

        self.flow_iface_assignment = np.vstack(flow_iface_assignment).T

    def generate_ifaces(self):
        self.prim_stations = np.empty_like(self.stations)
        self.station_ifaces = [[] for _ in self.stations]
        self.station_iface_assignment = [[] for _ in self.stations]
        ifaces = []
        size = self.env.grid_size
        # shared_ifaces = []

        def in_field(pos: np.ndarray[float]) -> bool:
            return pos[0] >= size[0] / 2\
                and pos[1] >= size[1] / 2\
                and pos[0] <= self.env.field_size[0] - size[0] / 2\
                and pos[1] <= self.env.field_size[1] - size[1] / 2

        iface_idx = 0
        for station_idx, station in enumerate(self.stations):
            prim_station = PrimStation(pos=station.pos, rot=None, size=size, fixed=True, incoming=None, outgoing=None, station=station)
            self.prim_stations[station_idx] = prim_station

        # Handle station by smallest average distance to avoid best iface spots being taken by stations with more available space
        for station_idx in np.argsort(np.sum(self.station_distances, axis=1)):
            prim_station = self.prim_stations[station_idx]
            num_flows = self.num_provided[station_idx] + self.num_received[station_idx]

            # If at upper or lower field border, try right and left space first. Else, try bottom and top first
            if prim_station.pos[1] <= size[1] / 2 or prim_station.pos[1] >= self.env.field_size[1] - size[1] / 2:
                vects = ((1, 0), (-1, 0), (0, 1), (0, -1))
            else:
                vects = ((0, 1), (0, -1), (1, 0), (-1, 0))

            for vect in vects:
                # if len(self.station_ifaces[station_idx]) >= 2: continue
                if num_flows <= 2 and len(self.station_ifaces[station_idx]) >= 2:
                    continue

                iface_pos = prim_station.pos + size * vect
                if not in_field(iface_pos):
                    continue
                if any(all(abs(n.pos - iface_pos) < size) for n in self.prim_stations):
                    continue

                if any(all(abs(i.pos - iface_pos) < size) for i in ifaces):
                    # Iface shared with other station
                    # iface = i
                    # shared_ifaces.append(iface)
                    continue
                else:
                    rot = np.rad2deg(np.arctan2(*vect)) + 90
                    iface = Iface(pos=iface_pos, rot=rot, id=iface_idx, size=size, fixed=True, incoming=None, outgoing=None, station=prim_station, is_source=None)
                    self.station_ifaces[station_idx].append(iface)
                    self.station_iface_assignment[station_idx].append(iface_idx)
                    if not in_field(prim_station.pos - size * vect):
                        self.center_ifaces.append(iface_idx)
                    iface_idx += 1

                ifaces.append(iface)

            if len(self.station_ifaces[station_idx]) < 2:
                raise Exception("No valid interfaces found")
        pass

    def generate_arcs(self):
        size = self.env.grid_size
        self.arcs = []
        arc_list = []
        num_ifaces = sum(len(l) for l in self.station_iface_assignment)

        for station1_idx, station1 in enumerate(self.stations):
            for station2_idx, station2 in enumerate(self.stations):
                if station1 == station2:
                    continue
                if not self.flows[station1_idx, station2_idx]:
                    continue

                for iface1 in self.station_ifaces[station1_idx]:
                    for iface2 in self.station_ifaces[station2_idx]:
                        arc = Arc(iface1, iface2, add_to_primitives=False, mover_size=size)
                        self.arcs.append(arc)
                        iface1_idx = iface1.id
                        iface2_idx = iface2.id
                        arc_idx = len(self.arcs) - 1
                        arc_list.append((iface1_idx, iface2_idx, arc_idx))
                        self.arc_flow[arc] = self.flows[station1_idx, station2_idx]
        self.arc_assignments = np.zeros((num_ifaces, num_ifaces), dtype=np.uint16)
        self.arc_assignments_inv = np.empty((len(arc_list), 2), dtype=np.uint16)
        for iface1_idx, iface2_idx, arc_idx in arc_list:
            self.arc_assignments[iface1_idx, iface2_idx] = arc_idx
            self.arc_assignments_inv[arc_idx] = (iface1_idx, iface2_idx)

        self.arcs = np.array(self.arcs)
        self.opposing_arcs = np.full_like(self.arcs, None)
        for arc1_idx, arc1 in enumerate(self.arcs):
            for arc2_idx, arc2 in enumerate(self.arcs):
                if arc1.head == arc2.tail and arc1.tail == arc2.head:
                    self.opposing_arcs[arc1_idx] = arc2_idx
                    continue

        def intersecting(arc1: Arc, arc2: Arc) -> bool:
            v1 = arc1.vect
            v2 = arc2.vect
            r = arc2.head.pos-arc1.tail.pos
            if np.cross(v1, v2) == 0:
                return False  # parallel or opposing - both are valid
            t, _ = np.linalg.solve(np.array([v1, v2]).T, r)
            return t > 0.01 and t < 0.99

        self.intersections = np.full((len(self.arcs), len(self.arcs)), 0.0)
        for arc1_idx, arc1 in enumerate(self.arcs):
            for arc2_idx, arc2 in enumerate(self.arcs):
                if arc1_idx != arc2_idx and intersecting(arc1, arc2):
                    flow = self.arc_flow[arc1] * self.arc_flow[arc2]
                    self.intersections[arc1_idx, arc2_idx] = flow
                    self.intersections[arc2_idx, arc1_idx] = flow
            # for arc2_idx, arc2 in enumerate(self.arcs[arc1_idx:]):
            #     self.intersections[arc1_idx, arc2_idx] = intersecting(arc1, arc2)

    def generate_station_map(self):
        self.station_map = np.empty_like(self.stations)

        def permute_station(a: np.ndarray) -> np.ndarray:
            out = np.meshgrid(*a, indexing='ij')
            out = np.moveaxis(out, 0, -1)
            out = out.reshape(-1, a.shape[0])
            return out[out[:, 0] != out[:, 1]]

        for station_idx in range(len(self.stations)):
            ifaces_idx = self.station_iface_assignment[station_idx]
            num_provided = self.num_provided[station_idx]
            num_received = self.num_received[station_idx]
            num_flows = num_provided + num_received
            num_ifaces = len(ifaces_idx)

            if num_ifaces == 2:
                tmp_map = np.zeros((2, num_flows), dtype=int)

                tmp_map[0] = [ifaces_idx[0]] * num_provided + [ifaces_idx[1]] * num_received
                tmp_map[1] = [ifaces_idx[1]] * num_provided + [ifaces_idx[0]] * num_received

            elif num_ifaces == 3:
                tmp_map = []
                center_idx = next(local_idx for local_idx, iface_idx in enumerate(ifaces_idx) if iface_idx in self.center_ifaces)
                center = ifaces_idx[center_idx]
                sides = np.delete(ifaces_idx, center_idx)

                tmp_map.append([sides[0]] * num_provided + [sides[1]] * num_received)
                tmp_map.append([sides[1]] * num_provided + [sides[0]] * num_received)

                if num_provided > 1:
                    provider_ifaces_permuted = permute_station(np.tile(sides, num_provided).reshape((num_provided, -1)))
                    receiver_ifaces_tiled = np.tile(center, (provider_ifaces_permuted.shape[0], num_received))
                    tmp_map.append(np.hstack([provider_ifaces_permuted, receiver_ifaces_tiled]))

                if num_received > 1:
                    receiver_ifaces_permuted = permute_station(np.tile(sides, num_received).reshape((num_received, -1)))
                    provider_ifaces_tiled = np.tile(center, (receiver_ifaces_permuted.shape[0], num_provided))
                    tmp_map.append(np.hstack([provider_ifaces_tiled, receiver_ifaces_permuted]))

            self.station_map[station_idx] = np.vstack(tmp_map)

    def generate_station_map_reduced(self):
        solution_arcs_idx = self.solution_arcs_arr.nonzero()[0]
        solution_ifaces_idx = self.arc_assignments_inv[solution_arcs_idx].T

        self.station_map = np.empty_like(self.stations)

        def permute_station_full(a: np.ndarray) -> np.ndarray:
            out = np.meshgrid(*a, indexing='ij')
            out = np.moveaxis(out, 0, -1)
            out = out.reshape(-1, a.shape[0])
            return out

        for station_idx in range(len(self.stations)):
            ifaces_idx = self.station_iface_assignment[station_idx]
            num_provided = self.num_provided[station_idx]
            num_received = self.num_received[station_idx]
            num_flows = num_provided + num_received
            num_ifaces = len(ifaces_idx)
            provider_idx = np.intersect1d(ifaces_idx, solution_ifaces_idx[0])[0]
            receiver_idx = np.intersect1d(ifaces_idx, solution_ifaces_idx[1])[0]

            if num_ifaces == 2:
                tmp_map = np.array([[provider_idx] * num_provided + [receiver_idx] * num_received], dtype=int)

            elif num_ifaces == 3:
                tmp_map = []
                free_iface_idx = next(idx for idx in ifaces_idx if idx not in [provider_idx, receiver_idx])

                if num_provided == 1 and num_received == 1:
                    tmp_map.append([provider_idx, receiver_idx])
                    tmp_map.append([free_iface_idx, receiver_idx])
                    tmp_map.append([provider_idx, free_iface_idx])

                if num_provided > 1:
                    provider_ifaces_permuted = permute_station_full(np.tile([provider_idx, free_iface_idx], num_provided).reshape((num_provided, -1)))
                    receiver_ifaces_tiled = np.tile(receiver_idx, (provider_ifaces_permuted.shape[0], num_received))
                    tmp_map.append(np.hstack([provider_ifaces_permuted, receiver_ifaces_tiled]))

                if num_received > 1:
                    receiver_ifaces_permuted = permute_station_full(np.tile([receiver_idx, free_iface_idx], num_received).reshape((num_received, -1)))
                    provider_ifaces_tiled = np.tile(provider_idx, (receiver_ifaces_permuted.shape[0], num_provided))
                    tmp_map.append(np.hstack([provider_ifaces_tiled, receiver_ifaces_permuted]))

            self.station_map[station_idx] = np.vstack(tmp_map)

    def generate_flowgraph(self, count_abort: int = np.inf, count_shared: bool = False):
        self.min_intersections = np.inf
        self.min_ifaces_shared = np.inf
        ifaces_shared = np.inf
        self.min_count = 0
        min_arcs = None
        count_since_last_improvement = 0
        count = 0

        self.num_perm = functools.reduce(operator.mul, map(len, self.station_map), 1)
        self.num_perm_digits = len(str(self.num_perm))
        LOGGER.log_info("Searching {0} permutations".format(self.num_perm))

        # Assignment are stacked [[Iface Idx for [Station Provided Flows, Station Received Flows]] for Stations], shape is (2 * NumFlows,)
        # ifaces = assignment[self.flow_iface_assignment] reorders array to [Flow Idx -> [Provider Iface Idx, Receiver Iface Idx]].T
        # arc = self.arc_assignments[*ifaces] looks up arcs connecting ifaces for each flow: [Flow Idx -> Arc Idx]
        # arcs_taken is [Arc Idx -> bool] <=> arc used in assignment for all arcs
        for assignment in itertools.product(*self.station_map):
            ifaces = np.hstack(assignment)[self.flow_iface_assignment]
            arc = self.arc_assignments[*ifaces]
            arcs_taken = np.full_like(self.arcs, False)
            arcs_taken[arc] = True

            num_intersections = arcs_taken @ self.intersections @ arcs_taken.T

            if count_shared:
                _, shared = np.unique(ifaces, return_counts=True)
                ifaces_shared = np.sum(shared > 1)

            if num_intersections < self.min_intersections or (num_intersections == self.min_intersections and ifaces_shared < self.min_ifaces_shared):
                self.min_intersections = num_intersections
                self.min_ifaces_shared = min(self.min_ifaces_shared, ifaces_shared)
                min_arcs = arcs_taken
                self.min_count = 1
                count_since_last_improvement = 0
                LOGGER.log_info("{1:{0}d}/{2:{0}d}: New lowest score {3}, shared {4}".format(self.num_perm_digits, count, self.num_perm, self.min_intersections, self.min_ifaces_shared))
            elif num_intersections == self.min_intersections and ifaces_shared == self.min_ifaces_shared:
                min_arcs = np.vstack([min_arcs, arcs_taken]).astype(bool)
                self.min_count += 1

            count += 1
            count_since_last_improvement += 1
            if count_since_last_improvement > count_abort:
                LOGGER.log_warning("Aborting after {0} permutations, no improvment since {1}".format(count, count_abort))
                break
            # if count % (self.num_perm // 10) == 0: LOGGER.log_cont("{0}/{1}: {2:.0%}".format(count, self.num_perm, count / self.num_perm))
        LOGGER.log_perf_mark("Search complete")

        if self.min_count == 1:
            self.min_frequency_arr = min_arcs.astype(int)
            self.arr_solution_score = np.array([1])
            self.solution_arcs_arr = min_arcs.astype(bool)
        else:
            self.min_frequency_arr = np.sum(min_arcs, axis=0) / self.min_count
            self.arr_solution_score = min_arcs @ self.min_frequency_arr
            uniques, count = np.unique(self.arr_solution_score, return_counts=True)
            best_sol = np.argmax(uniques)
            self.solution_arcs_arr = min_arcs[best_sol].astype(bool)
            if count[best_sol] == 2 and all(self.opposing_arcs[min_arcs[0]] != None) and all(min_arcs[1, self.opposing_arcs[min_arcs[0]].astype(int)]):
                LOGGER.log_warning("Symmetric ideal solution found. Direction of travel can be inverted.")
                self.sol.reversible = True
            elif count[best_sol] > 1:
                LOGGER.log_warning("{0} ideal solutions found. Using first.".format(count[best_sol]))

    def generate_flow(self):
        solution_arcs = list(self.arcs[self.solution_arcs_arr])
        solution_ifaces = set()
        for arc in solution_arcs:
            arc.tail.is_source = True
            arc.head.is_source = False
            arc.add_to_primitives()
            solution_ifaces.add(arc.tail)
            solution_ifaces.add(arc.head)
        solution_ifaces = list(solution_ifaces)

        self.sol.flow = Network(primitives=list(self.prim_stations) + solution_ifaces, arcs=solution_arcs)
