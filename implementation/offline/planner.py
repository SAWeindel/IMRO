import numpy as np

from definition import *
from tools.log import LOGGER

from .flow_generator import FlowGenerator
from .placement_optimizer import PlacementOptimizer
from .sequence_generator import SequenceGenerator


class Planner:
    def __init__(self, env: Environment, sol: Solution = None, init_flow: Network = None, init_network: Network = None):
        self.env = env

        self.sol = sol or Solution(self.env)
        if init_flow:
            self.sol.flow = init_flow
        if init_network:
            self.sol.networks = [init_network]

        self.mapper = FlowGenerator(self.env, self.sol)
        self.optimizer: PlacementOptimizer = None
        self.sequence_generator: SequenceGenerator = None

    def solve(self, use_alltoall: bool, split_ifaces: bool, use_merge_split: bool, max_length: int | None, reverse: bool):
        if not self.sol.networks:
            if not self.sol.flow:
                self.generate_flow(use_alltoall=use_alltoall, split_ifaces=split_ifaces)
        self.initialize_network(use_merge_split=use_merge_split, max_length=max_length)
        self.optimize_network()
        if reverse and self.sol.reversible:
            self.reverse_network()
        self.post_process()
        self.generate_graph()
        self.generate_item_sequences()
        return self.export_solution()

    def generate_flow(self, use_alltoall: bool, split_ifaces: bool = False):
        self.mapper.run(use_alltoall=use_alltoall, split_ifaces=split_ifaces)

    def initialize_network(self, use_merge_split: bool, max_length: int | None):
        network = self.sol.flow.copy()

        network.generate_primitives_iface_arcs(size=self.sol.env.grid_size)
        network.generate_proto_paths(flows_dict=self.mapper.flows_dict)

        network.generate_primitives_cross(size=self.sol.env.grid_size, use_merge_split=use_merge_split)
        network.generate_primitives_split_merge(size=self.sol.env.grid_size)

        if max_length is not None:
            network.generate_corners_maxlength(size=self.sol.env.grid_size, max_length=max(*self.sol.env.grid_size) * max_length)

        network.assign_ids(reassign=True)

        self.sol.networks = [network]

    def reverse_network(self):
        self.sol.networks.append(self.sol.networks[-1].copy_reverse())

    def use_provided_flow(self, flow: Network):
        self.sol.flow = flow

    def use_checkpoint_network(self, network: Network):
        self.sol.networks = [network]

    def optimize_network(self, draw_vid: bool = True, show_vid_freq: int | None = 0, checkpoint_path: str | None = "./output/"):
        self.optimizer = PlacementOptimizer(self.env, self.sol)
        try:
            self.optimizer.run(draw_vid=draw_vid, show_vid_freq=show_vid_freq, checkpoint_path=checkpoint_path)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            raise e

    def post_process(self):
        self.sol.networks[-1].post_process()
        self.sol.networks[-1].generate_primitives_waiting(size=self.sol.env.grid_size)
        self.sol.networks[-1].generate_sections()

    def generate_graph(self):
        self.sol.graph = self.network_to_graph(self.sol.networks[-1])

    def generate_item_sequences(self):
        if self.sol.graph is None:
            raise Exception("Graph must be generated before sequences can be assigned")
        self.sequence_generator = SequenceGenerator(graph=self.sol.graph)
        self.sol.graph.item_sequences = self.sequence_generator.generate_sequences()

    def export_solution(self, path: str | None = "./output/", pickle_filename: str | None = 'sol.p') -> Solution:
        if path:
            if self.sol.flow is not None:
                self.sol.drawer.save_network_img(source=self.sol.flow, text="Flow Network", filename=path+"sol_a_flow.png", draw_env=True, draw_forces=False)
                LOGGER.to_json(self.sol.flow.to_dict(), path=path, name="sol_a_flow")
            if self.sol.networks is not None:
                self.sol.drawer.save_network_img(source=self.sol.networks[0], text="Initial Network", filename=path+"sol_b_init.png", draw_env=True, draw_forces=False)
                LOGGER.to_json(self.sol.networks[0].to_dict(), path=path, name="sol_b_init")
            if len(self.sol.networks) > 2:
                self.sol.drawer.save_network_img(source=self.sol.networks[-2], text="Optimized Network", filename=path+"sol_c_optim.png", draw_env=True, draw_forces=False)
                LOGGER.to_json(self.sol.networks[-2].to_dict(), path=path, name="sol_c_optim")
            self.sol.drawer.save_network_img(source=self.sol.networks[-1], text="Post-Processed Network", filename=path+"sol_d_post.png", draw_env=True, draw_forces=False)
            LOGGER.to_json(self.sol.networks[-1].to_dict(), path=path, name="sol_d_post")
            if self.sol.graph is not None:
                self.sol.drawer.save_network_img(source=self.sol.graph, text="Graph", filename=path+"sol_e_graph.png")
            if pickle_filename:
                self.sol.to_pickle(filename=path+pickle_filename)

        if self.optimizer is not None and self.optimizer.drawer is not None:
            self.optimizer.drawer.frame_network_vid(network=self.sol.networks[-1], text="Post-Processing completed", frame_count=5)
            self.optimizer.drawer.close_network_vid()

        return self.sol

    def network_to_graph(self, network: Network) -> Graph:
        nodes: list[Node] = [Node(pos=p.pos, id=p.id) for p in network.primitives]
        edges: list[Edge] = []
        paths: dict[tuple[Node, Node], EdgePath] = {}
        sections: dict[tuple[Node, Node], EdgeSection] = {}

        node_assignment: dict[Primitive, Node] = {p: n for p, n in zip(network.primitives, nodes)}
        edge_assignment: dict[Arc, Edge] = {}
        section_assignment: dict[Section, EdgeSection] = {}

        for prim, node in zip(network.primitives, nodes):
            if len(prim.incoming) > 1:
                time_clear_fact = 0.75
            else:
                time_clear_fact = 0
                for i in prim.incoming:
                    for o in prim.outgoing:
                        if np.abs(i.rot - o.rot) > 45:
                            time_clear_fact = 1
            node.time_clear *= time_clear_fact

            edge_type = EdgeType.MOVE
            if type(prim) is PrimStation:
                node.station = prim.station
                edge_type = EdgeType.LEAVE_STATION
            elif type(prim) is Iface and prim.is_sink:
                edge_type = EdgeType.ENTER_STATION
            elif type(prim) in (Merge, Cross, Alltoall):
                node.can_wait = False

            for arc in prim.outgoing:
                node_head = node_assignment[arc.head]
                path_goals = [p.goal.station for p in arc.paths]
                edge_type = EdgeType.MOVE_NOWAIT if arc.min_spaces <= 1 and node_head.can_wait == False else EdgeType.MOVE
                edge = Edge(tail=node, head=node_head, edge_type=edge_type, path_goals=path_goals)
                edge_assignment[arc] = edge
                edges.append(edge)
                node.edges.append(edge)
                node_head.edges_incoming.append(edge)

        for path in network.paths.values():
            path_edges = [edge_assignment[a] for a in path.arcs]
            items = path.items.copy()
            edge_path = EdgePath(edges=path_edges, items=items)
            paths[(edge_path.start, edge_path.goal)] = edge_path

        for section in network.sections.values():
            section_edges = [edge_assignment[a] for a in section.arcs]
            items = section.items.copy()
            flows_dict = section.flows_dict.copy()
            edge_section = EdgeSection(edges=section_edges, items=items, type=section.type, flows_dict=flows_dict)
            sections[(edge_section.start, edge_section.goal)] = edge_section
            section_assignment[section] = edge_section

        for section, edge_section in section_assignment.items():
            edge_section.sections_prev = tuple(section_assignment[s] for s in section.sections_prev)
            edge_section.sections_next = tuple(section_assignment[s] for s in section.sections_next)

        return Graph(nodes=nodes, edges=edges, paths=paths, sections=sections)
