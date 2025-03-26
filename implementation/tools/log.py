from __future__ import annotations
import os
import time
import json


class Log:
    generate_logs = True
    verbose = 3  # [0: No prints, 1: only notes, 2: notes and errors, 3: notes, errors and warnings]

    def __init__(self, path: str | None = "./output/"):
        self.path = path
        self.tics: list[tuple[str, int]] = [("Init", time.perf_counter_ns())]
        self.num_errors = 0
        self.num_warnings = 0
        self.modified = set()
        self.last_log_cont = False

    def reset(self):
        self.tics: list[tuple[str, int]] = [("Init", time.perf_counter_ns())]
        self.num_errors = 0
        self.num_warnings = 0
        self.modified = set()
        self.last_log_cont = False

    def log_print_data(self, name: str, data, filetype: str | None = "txt", path=None):
        if not self.generate_logs:
            return
        if path is None:
            path = self.path
        with open("{0}{1}.{2}".format(path, name, filetype), 'w') as f:
            f.write(data)
        if self.verbose > 0:
            print("Logged {0}.{1}".format(name, filetype))

    def log_print_cont(self, name: str, data, filetype: str | None = "txt"):
        if name not in self.modified:
            self.log_print_line(name=name, data=data, filetype=filetype, end="")
            return
        with open("{0}{1}.{2}".format(self.path, name, filetype), 'r+b') as f:
            f.seek(-1, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            f.write(bytes(data, 'utf-8'))
            f.truncate()

    def log_print_line(self, name: str, data, end: str | None = "\n", filetype: str | None = "txt"):
        mode = 'a' if name in self.modified else 'w'
        self.modified.add(name)
        with open("{0}{1}.{2}".format(self.path, name, filetype), mode) as f:
            f.write(data + end)

    def log_print_iter(self, name: str, data: list, filetype: str | None = "txt"):
        if not self.generate_logs:
            return
        out = "LOG {0}:\n".format(name)
        for e in data:
            out += "{0}\n".format(e)
        self.log_print_data(name, out, filetype)

    def log_cont(self, text: str):
        self.log_print_cont("log", "INFO  {0}".format(text))
        if self.verbose > 0:
            print("\rINFO  {0}".format(text), end="")
        self.last_log_cont = True

    def log_info(self, text: str):
        line = "{0}INFO  {1}".format("\n" if self.last_log_cont else "", text)
        self.log_print_line("log", line)
        if self.verbose > 0:
            print(line)
        self.last_log_cont = False

    def log_error(self, text: str):
        self.num_errors += 1
        line = "{0}!ERR! {1}".format("\n" if self.last_log_cont else "", text)
        self.log_print_line("log", line)
        if self.verbose > 1:
            print(line)
        self.last_log_cont = False

    def log_warning(self, text: str):
        self.num_warnings += 1
        line = "{0}WARN  {1}".format("\n" if self.last_log_cont else "", text)
        self.log_print_line("log", line)
        if self.verbose > 2:
            print(line)
        self.last_log_cont = False

    def log_empty(self):
        line = "{0}".format("\n" if self.last_log_cont else "")
        self.log_print_line("log", line)
        if self.verbose > 0:
            print(line)
        self.last_log_cont = False

    def log_num_warning(self, reset: bool = False):
        if self.verbose > 0:
            print("{0} errors logged".format(self.num_errors)) if self.num_errors > 0 else print("No errors logged")
            print("{0} warnings logged".format(self.num_warnings)) if self.num_warnings > 0 else print("No warnings logged")
        if reset:
            self.num_errors = 0
            self.num_warnings = 0

    def log_perf(self):
        out = "Overall performance times\n"
        for i, (name, t) in enumerate(self.tics[:-1]):
            out += "      {0}: {1:.3f} ms\n".format(name, (self.tics[i + 1][1] - t) / 1_000_000)
        self.log_info(out)

    def log_perf_start(self, next: str):
        tic = time.perf_counter_ns()
        text = "PERF  Starting {0}".format(next)
        self.log_info(text)
        self.tics.append((next, tic))

    def log_perf_end(self):
        assert len(self.tics)
        tic = time.perf_counter_ns()
        text = "PERF  {0} took {1:.3f} ms".format(self.tics[-1][0], (tic-self.tics[-1][1]) / 1_000_000)
        self.log_info(text)
        last = "END: " + self.tics[-1][0]
        self.tics.append((last, tic))

    def log_perf_mark(self, next: str):
        tic = time.perf_counter_ns()
        text = "PERF  {0} took {1:.3f} ms\n      Starting {2}".format(self.tics[-1][0], (tic-self.tics[-1][1]) / 1_000_000, next)
        self.log_info(text)
        self.tics.append((next, tic))

    def log_sol(self, sol=None):
        if not self.generate_logs:
            return
        if sol is not None:
            self.sol = sol
        if self.sol.graph is not None:
            self.log_nodes()

    def log_nodes(self):
        if not self.generate_logs:
            return
        out = "Grid log:\n{0:d} Providers, {1:d} Receivers" \
            .format(len(self.sol.env.providers), len(self.sol.env.receivers))
        for node in self.sol.graph.nodes:
            out += "\nNode at [{0:4.0f},{1:4.0f}], {2} edges, {3} incoming, station is {4}\nNeighbors:\n" \
                .format(node.pos[0], node.pos[1], len(node.edges), len(node.edges_incoming), node.station)
            out += "Edges:\n"
            for edge in node.edges:
                out += "\t{0}\n".format(repr(edge))
            out += "Edges incoming:\n"
            for edge in node.edges_incoming:
                out += "\t{0}\n".format(repr(edge))
        self.log_print_data("nodes", out)

    def log_seq(self, seq=None):
        if not self.generate_logs:
            return
        if seq is None:
            return
        self.seq = seq
        self.log_events()
        self.log_actions()

    def log_events(self):
        if not self.generate_logs:
            return
        out = "Sequence events log:\n{0:d} Movers, {1:d} Providers, {2:d} Receivers, {3:.1f} seconds\n" \
            .format(self.seq.sce.mover_num, len(self.seq.sol.env.providers), len(self.seq.sol.env.receivers), self.seq.sce.time_max)
        for (station, events) in self.seq.events.items():
            out += "\n{0}:\n".format(repr(station))
            for event in events:
                out += "\t{0}\n".format(repr(event))
        self.log_print_data("events", out)

    def log_tasks(self):
        if not self.generate_logs:
            return
        out = "Sequence task log:\n{0:d} Movers, {1:d} Providers, {2:d} Receivers, {3:.1f} seconds\n" \
            .format(self.seq.sce.mover_num, len(self.seq.sol.env.providers), len(self.seq.sol.env.receivers), self.seq.sce.time_max)
        for mover, tasks in self.seq.tasks.items():
            out += "\n{0}:\n".format(repr(mover))
            for task in tasks:
                out += "\t{0}\n".format(repr(task))
        self.log_print_data("tasks", out)

    def log_actions(self):
        if not self.generate_logs:
            return
        out = "Sequence actions log:\n{0:d} Movers, {1:d} Providers, {2:d} Receivers, {3:.1f} seconds" \
            .format(self.seq.sce.mover_num, len(self.seq.sol.env.providers), len(self.seq.sol.env.receivers), self.seq.sce.time_max)
        for mover in self.seq.movers:
            i_action = 0
            out += "\nMover {0}, init {1}:\n".format(mover, mover.init_node)
            for path in mover.paths:
                out += "\t{0}\n".format(repr(path))
                while i_action < len(mover.actions):
                    action = mover.actions[i_action]
                    out += "\t{0:3d}: {1}\n".format(i_action, repr(action))
                    i_action += 1
                    if action.edge.head == path.goal:
                        break
        self.log_print_data("actions", out)

    def log_reserved(self):
        if not self.generate_logs:
            return
        counts = {}
        matched = {}
        for node in self.sol.graph.nodes:
            for step, v in node.reserved_incoming.items():
                if step not in matched:
                    matched[step] = {}
                matched[step][node] = (v, "-" * 80)

                if v not in counts:
                    counts[v] = 1
                else:
                    counts[v] += 1

            for step, v in node.reserved_outgoing.items():
                if step not in matched:
                    matched[step] = {}
                if node in matched[step]:
                    matched[step][node] = (matched[step][node][0], v)
                elif step-1 in matched and node in matched[step-1] and matched[step-1][node][0] == v:
                    matched[step-1][node] = (matched[step-1][node][0], v)
                    matched[step][node] = (matched[step-1][node][0], v)
                else:
                    matched[step][node] = ("-" * 80, v)

                if v not in counts:
                    counts[v] = 1
                else:
                    counts[v] += 1

        out = "{0} unique actions\n\nIrregular actions:\n".format(len(counts))
        for v, c in sorted(counts.items(), reverse=True, key=lambda i: i[1]):
            if c != 2:
                out += "{0:5d}: {1}\n".format(c, v)
        out += "\n\n"
        for step, nodes in sorted(matched.items()):
            out += "Step {0}: {1}\n".format(step, len(nodes))
            for node, (v1, v2) in nodes.items():
                out += "\t{0}: {1:<80} -> {2}\n".format(node, str(v1), str(v2))
                # out += "{0:<80} -> {1}\n".format(str(v1), str(v2))
        self.log_print_data("reserved", out)

    def to_json(self, x: dict, name: str = None, path: str = "./output/", filetype: str | None = "json") -> str | None:
        out = self._json_dumps_recursive(x)
        if name:
            self.log_print_data(name=name, data=out, filetype=filetype, path=path)
        else:
            return out

    def _json_dumps_recursive(self, x: dict | list, depth: int = 0) -> str:
        if type(x) is dict and '_nosplit' in x:
            x.pop('_nosplit')
            return json.dumps(x)

        if type(x) is list and len(x) > 0:
            out = "["
            for v in x:
                out += "\n{0}{1},".format('\t'*depth, self._json_dumps_recursive(v, depth=depth+1))
            out = out[:-1] + "\n{0}]".format('\t'*depth)
            return out

        if type(x) is dict:
            out = "{"
            for k, v in x.items():
                out += "\n{0}\"{1}\": {2},".format('\t'*depth, k, self._json_dumps_recursive(v, depth=depth+1))
            out = out[:-1] + "\n{0}}}".format('\t'*depth)
            return out

        return json.dumps(x)


LOGGER = Log()
