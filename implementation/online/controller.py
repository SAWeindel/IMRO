from definition import *
from tools.log import LOGGER


class Controller:
    def __init__(self, sol: Solution):
        self.sol = sol
        self.graph = sol.graph

    def setup(self, iface: StationInterface, movers: list[Mover]):
        self.iface = iface
        self.movers = movers

        self.dep_releases: set[Mover] = set()

        self.get_init_pos()

    def get_init_pos(self):
        path_lists = [[p for pk, p in self.graph.paths.items() if pk[1].station in self.sol.env.providers],
                      [p for pk, p in self.graph.paths.items() if pk[1].station in self.sol.env.receivers]]

        movers = iter(self.movers)
        mover: Mover = next(movers)
        for paths in path_lists:
            for i in range(0, max(len(p.edges) for p in paths)):
                for path in paths:
                    if len(path.edges) < i:
                        continue
                    node = path.nodes[-(i + 1)]
                    if not node.can_wait or node.mover_reserve_wait:
                        continue

                    mover.init_node = node
                    node.mover_reserve_wait = mover
                    if i == 0:  # at station
                        mover.state = MoverState.TO_STATION
                    else:  # on path
                        mover.paths = [path]
                        mover.state = MoverState.TO_INIT

                    mover: Mover = next(movers, None)
                    if not mover:
                        return

        raise Exception("Not enough nodes available to init place all movers")

    def get_actions(self, mover: Mover, time_planning: float):
        time_start = max(mover.next_time_to_plan, time_planning)
        move_rem = mover.move_rem

        LOGGER.log_print_line("simulation", "{0:6.2f} Planning Mover {1} @{2}, {3}, item {4}, next action {5:6.2f}, for {6}".format(time_planning, mover, mover.get_loc(time_planning), mover.state.name, mover.item, time_start, move_rem))

        actions, has_safe, blocking_type, blocking_var, new_sections, reached_station = self.route_recursive(mover=mover, move_rem=move_rem, time_start=time_start)

        time_next = max(actions[-1].time_goal, time_start) if actions else time_start
        LOGGER.log_print_line("simulation", "\tPlanned until {0:6.2f} {1}\n\t\t{2}\n\t\t{3}"
                              .format(time_next, actions[-1].node_goal if actions else "", actions or "No Actions", new_sections or "No new Sections"))

        self.set_mover_state(mover=mover, actions=actions, blocking_type=blocking_type, time=time_start)
        self.set_continuation(mover=mover, blocking_type=blocking_type, blocking_var=blocking_var, time_next=time_next)

        if new_sections:
            for section in new_sections:
                node_start = section.start
                dep_releases = node_start.commit(mover=mover, section=section, time=time_planning)
                self.dep_releases |= dep_releases
                if dep_releases:
                    LOGGER.log_print_line("simulation", "\tSection {0} dep release {1}".format(section, [m.id for m in dep_releases]))
            mover.current_section = new_sections[-1]

        if reached_station:
            dep_releases = reached_station.commit_station(mover=mover, time=time_planning)
            self.dep_releases |= dep_releases
            if dep_releases:
                LOGGER.log_print_line("simulation", "\tStation {0} dep release {1}".format(reached_station, [m.id for m in dep_releases]))

        if actions:
            mover.actions.extend(actions)
            mover.next_time_to_plan = max(mover.next_time_to_plan, actions[-1].time_goal)

            for action in actions:
                node_start = action.node_start
                node_start.reserve_pass(mover=mover, action_time_start=action.time_start)
            node_goal = actions[-1].node_goal
            node_goal.reserve_wait(mover=mover)

            dep_releases = mover.get_blocked_dependencies_copy()
            self.dep_releases |= dep_releases
            if dep_releases:
                LOGGER.log_print_line("simulation", "\tMover {0} dep release {1}".format(mover, [m.id for m in dep_releases]))

        while self.dep_releases:
            release_queue = sorted(self.dep_releases, key=lambda m: m.time_last_action)
            blocked_mover = release_queue[0]
            LOGGER.log_print_line("simulation", "\n{0:6.2f} Dep release queue: {1}".format(time_planning, [m.id for m in release_queue]))
            self.dep_releases.remove(blocked_mover)
            self.get_actions(mover=blocked_mover, time_planning=time_planning)

    def route_recursive(self, mover: Mover, move_rem: tuple[Edge], time_start: float, depth: int = 0)\
            -> tuple[list[Action], bool, BlockType, Mover | Node | float | None, list[EdgeSection], Node | None]:
        # -> actions, has_safe, blocking_type, blocking_var, new_sections, reached_station
        actions: list[Action] = []
        safe_idx: int = None
        time_next = time_start
        node_next: Node = None

        blocking_type: BlockType = None
        blocking_var: Mover | Node | float = None
        new_sections = []

        reached_station = None

        def print_safe_node() -> Node | None:
            return actions[safe_idx].node_goal if safe_idx else None

        if not move_rem:
            node_next = mover.target_node
            LOGGER.log_print_line("simulation", "\tRouting from {0}@{1:6.2f}, depth {2}".format(node_next, time_start, depth))
        else:
            LOGGER.log_print_line("simulation", "\tRouting {0}@{1:6.2f} -> {2}, depth {3}".format(move_rem[0].tail, time_start, move_rem[-1].head, depth))

        for idx, edge in enumerate(move_rem):
            node_next = edge.head
            action = Action(time_start=time_next, edge=edge, mover=mover)

            if node_next.mover_reserve_wait:  # Check if next node blocked for waiting
                blocking_type = BlockType.MOVER
                blocking_var = node_next.mover_reserve_wait
                LOGGER.log_print_line("simulation", "\t\tBlocked by Mover {0} @{1} to enter {2} at {3:6.2f}, prev safe {4}".format(blocking_var, blocking_var.get_loc(time_start), node_next, time_next, print_safe_node()))
                break
            elif node_next.time_next_free > action.time_goal:  # check if next node would be reached too early
                time_next = node_next.time_next_free - action.duration + 0.001
                del_actions = actions[safe_idx + 1:] if safe_idx else actions
                for a in del_actions:
                    time_next -= a.duration

                blocking_type = BlockType.TIME
                blocking_var = time_next
                LOGGER.log_print_line("simulation", "\t\tFor {0}: {1:6.2f} + {2:4.2f} = {3:6.2f} > {4:6.2f}".format(repr(action),
                                      node_next.time_next_free - node_next.time_clear, node_next.time_clear, node_next.time_next_free, action.time_goal))
                LOGGER.log_print_line("simulation", "\t\tWaiting for {0:6.2f} to enter {1} at {2:6.2f}, prev safe {3}".format(time_next, node_next, node_next.time_next_free, print_safe_node()))
                break

            if node_next.station is not None and mover.item is not None:
                if node_next.station_sequence.allowed(mover=mover, node=node_next):
                    reached_station = node_next
                    LOGGER.log_print_line("simulation", "\t\tReached Station {0} at {1:6.2f}".format(node_next, action.time_goal))
                else:
                    blocking_type = BlockType.ITEM
                    blocking_var = node_next
                    LOGGER.log_print_line("simulation", "\t\tBlocked by item seq to enter Station {0} at {1:6.2f}, prev safe {2}, next item {3}"
                                          .format(node_next, time_next, print_safe_node(), node_next.station_sequence.next_item))
                    break

            time_next = action.time_goal
            actions.append(action)
            if node_next.can_wait:
                safe_idx = idx

        has_safe = safe_idx is not None
        blocked = blocking_type is not None

        if has_safe or reached_station is not None or blocked:
            actions = actions[:safe_idx + 1] if has_safe else []
            return actions, has_safe, blocking_type, blocking_var, [], reached_station

        next_sections = node_next.get_valid_sections(mover=mover)
        if not next_sections:
            LOGGER.log_print_line("simulation", "\t\tBlocked by item seq to enter {0} at {1:6.2f}, prev safe {2}".format(node_next, time_next, print_safe_node()))
            blocking_type = BlockType.ITEM
            blocking_var = node_next
            return [], False, blocking_type, blocking_var, [], None

        next_section = None
        for section in next_sections:
            section_actions, section_has_safe, section_blocking_type, section_blocking_var, section_new_sections, reached_station = self.route_recursive(mover=mover, move_rem=section.edges, time_start=time_next, depth=depth+1)
            if section_has_safe:
                next_section = section
                actions.extend(section_actions)
                new_sections = [section] + section_new_sections
                return actions, True, blocking_type, blocking_var, new_sections, reached_station

        if not next_section:
            LOGGER.log_print_line("simulation", "\t\tNo safe continuation after {0} at {1:6.2f}, prev safe {2}".format(node_next, time_next, print_safe_node()))
            blocking_type = BlockType.NODE
            blocking_var = node_next

        return [], False, blocking_type, blocking_var, [], None

    def set_mover_state(self, mover: Mover, actions: list[Action], blocking_type: BlockType | None, time: float):
        if blocking_type is BlockType.MOVER:
            state = MoverState.TO_BL_PATH if actions else MoverState.BL_PATH
        elif blocking_type is BlockType.ITEM:
            state = MoverState.TO_BL_ITEM if actions else MoverState.BL_ITEM
        elif blocking_type is BlockType.NODE:
            state = MoverState.TO_BL_NODE if actions else MoverState.BL_NODE
        elif blocking_type is BlockType.TIME:
            state = MoverState.TO_BL_TIME if actions else MoverState.BL_TIME
        else:
            state = MoverState.TO_STATION if actions[-1].node_goal.station else MoverState.TO_GOAL
        mover.set_state(state, time=time)

    def set_continuation(self, mover: Mover, blocking_type: BlockType, blocking_var: Mover | Node | float, time_next: float):
        if blocking_type is BlockType.MOVER:
            mover_blocking: Mover = blocking_var
            mover_blocking.add_blocked_dependency(blocked_mover=mover)
        elif blocking_type is BlockType.ITEM:
            node_blocking: Node = blocking_var
            node_blocking.add_dependency(blocked_mover=mover)
        elif blocking_type is BlockType.TIME:
            time_next_free: float = blocking_var
            mover.next_time_to_plan = time_next_free
        elif blocking_type is BlockType.NODE:
            mover.next_time_to_plan = time_next + 0.1
