import bisect
import json
from typing import Dict
import networkx as nx
import xml.etree.ElementTree as ET

from pathlib import Path
from scipy.sparse.coo import coo_matrix
import utils


class BaseGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()

    def load_nodes_and_edges(self, graph: nx.DiGraph):
        self.add_nodes_from(graph.nodes(data=True))
        self.add_edges_from(graph.edges(data=True))

    def load(self, path: str):
        extension = utils.get_file_extension(path)

        if extension == "xml":
            self._load_xml(path=path)
        elif extension == "json":
            self._load_json(path=path)
        else:
            raise Exception(
                "Error when loading file {}: extension {} not supported.".
                format(path, extension))

    def parse(self, string: str, extension: str):
        if extension == "xml":
            self._load_xml(string=string)
        elif extension == "json":
            self._load_json(string=string)
        else:
            raise Exception(
                "Error when loading string: extension {} not supported.".
                format(extension))

    def _load_xml(self, path: str = None, string: str = None):
        pass

    def _load_json(self, path: str = None, string: str = None):
        if path is None and string is None:
            return

        if path is None:
            data = json.loads(string)
        else:
            with open(path, mode="r") as f:
                data = json.load(f)

        graph = nx.node_link_graph(data)

        self.load_nodes_and_edges(graph)
        self._load_other_elements_from_json(data)

    def _load_other_elements_from_json(self, data: dict):
        pass

    def save(self, path: str):
        file = Path(path)
        utils.create_parent_folders(file)

        data = self._write_data()

        with open(file, mode="w") as f:
            json.dump(data, f, indent=2)

    def write(self) -> str:
        return json.dumps(self._write_data(), indent=2)

    def _write_data(self) -> dict:
        data = nx.node_link_data(self)
        data = self._write_other_elements_in_data(data)
        return data

    def _write_other_elements_in_data(self, data: dict) -> dict:
        return data


class MulvalAttackGraph(BaseGraph):
    def __init__(self):
        super().__init__()

    def _load_xml(self, path: str = None, string: str = None):
        if path is None and string is None:
            return

        # Parse the file or the string
        if path is None:
            tree = ET.ElementTree(ET.fromstring(string))
        else:
            tree = ET.parse(path)

        root = tree.getroot()
        nodes = root.findall(path="vertices/vertex")
        edges = root.findall(path="arcs/arc")

        # Add nodes to the instance
        for i, node in enumerate(nodes):
            fact = node.find("fact").text
            metric = int(node.find("metric").text)
            type_ = node.find("type").text

            shape = "box"
            if type_ == "AND":
                shape = "oval"
            elif type_ == "OR":
                shape = "diamond"

            self.add_node(i,
                          fact=fact,
                          metric=metric,
                          type=type_,
                          label="{}: {}".format(i, fact),
                          shape=shape)

        # Add edges to the instance
        for edge in edges:
            # MulVAL starts the indexing of the nodes at 1 but we start at 0
            src = int(edge.find("src").text) - 1
            dst = int(edge.find("dst").text) - 1

            # Source and destination nodes seem inverted in the attack graph
            # files generated by MulVAL
            self.add_edge(dst, src)


class AttackGraph(BaseGraph):
    def __init__(self):
        super().__init__()

        self.propositions = {}
        self.exploits = {}
        self.initial_node = None
        self.final_node = None

    def copy(self, as_view=False):
        graph = super().copy(as_view=as_view)

        new_graph = AttackGraph()
        new_graph.load_nodes_and_edges(graph)
        new_graph.propositions = self.propositions.copy()
        new_graph.exploits = self.exploits.copy()
        new_graph.initial_node = self.initial_node
        new_graph.final_node = self.final_node

        return new_graph

    def get_pruned_graph(self, ids_exploits_to_keep: list):
        # Copy this attack graph
        new_graph = self.copy()

        # Remove the edges corresponding to the exploits to remove
        edges_to_remove = []
        for src, dst, id_exploit in new_graph.edges(data="id_exploit"):
            if id_exploit not in ids_exploits_to_keep:
                edges_to_remove.append((src, dst))
        new_graph.remove_edges_from(edges_to_remove)

        has_removed_nodes = True
        while has_removed_nodes:
            nodes_to_remove = []
            for i, ids_propositions in new_graph.nodes(
                    data="ids_propositions"):
                # The initial node can't be removed
                if i == 0:
                    continue

                # The nodes that have no predecessors must be removed
                if len(list(new_graph.predecessors(i))) == 0:
                    nodes_to_remove.append(i)

                # The nodes that have no successors (except the final node)
                # must be removed
                if i != new_graph.final_node and len(
                        list(new_graph.successors(i))) == 0:
                    nodes_to_remove.append(i)

            new_graph.remove_nodes_from(nodes_to_remove)
            has_removed_nodes = len(nodes_to_remove) > 0

        return new_graph

    def get_node_mapping(self) -> dict:
        ids_nodes = list(self.nodes)
        return dict([(id, i) for i, id in enumerate(ids_nodes)])

    def get_proposition_mapping(self) -> dict:
        ids_propositions = [*self.propositions]
        return dict([(ids_propositions[i], i)
                     for i in range(len(self.propositions))])

    def compute_adjacency_matrix(self,
                                 keep_directed: bool = True) -> coo_matrix:
        if keep_directed:
            network = self
        else:
            network = nx.Graph(self)
        return nx.adjacency_matrix(network)

    def set_final_node_id(self):
        self.final_node = max(
            [(i, len(ids_propositions))
             for i, ids_propositions in self.nodes(data="ids_propositions")],
            key=lambda node: node[1])[0]

    def get_nodes_layers(self) -> Dict[int, int]:
        nodes_layers = dict()
        n_initial_propositions = len(self.nodes[0]["ids_propositions"])
        for id_node, ids_propositions in self.nodes(data="ids_propositions"):
            layer = len(ids_propositions) - n_initial_propositions
            nodes_layers[id_node] = layer
        return nodes_layers

    def _load_xml(self, path: str = None, string: str = None):
        mag = MulvalAttackGraph()
        mag._load_xml(path, string)
        self._load_from_mulval_attack_graph(mag)

    def _load_from_mulval_attack_graph(self, mag: MulvalAttackGraph):
        ids_initial_propositions = []
        ids_edges = []

        for i, node in mag.nodes(data=True):
            if node["type"] == "LEAF":
                # The node is a LEAF which corresponds to a proposition that
                # is true at the beginning
                ids_initial_propositions.append(i)
                self.propositions[i] = node["fact"]
            elif node["type"] == "OR":
                # The node is an OR which corresponds to a proposition that
                # is false at the beginning
                self.propositions[i] = node["fact"]
            else:
                # The node is an AND which corresponds to an edge
                ids_edges.append(i)
                # This node also corresponds to an exploit on a particular
                # machine
                self.exploits[i] = node["fact"]

        # Create the initial node
        initial_node = (0, {"ids_propositions": ids_initial_propositions})
        self.add_nodes_from([initial_node])

        # Fill the graph
        self._fill_graph_recursively(mag, initial_node, ids_edges)

        # Set the initial and final nodes
        self.initial_node = 0
        self.set_final_node_id()

    def _fill_graph_recursively(self, mag: MulvalAttackGraph, node: tuple,
                                ids_edges: list):
        current_ids_propositions = node[1]["ids_propositions"]
        # Loop through all the edges
        for id_edge in ids_edges:
            # Check if this edge can be used depending on the current true
            # propositions
            is_possible = True
            for id_required in mag.predecessors(id_edge):
                is_possible &= id_required in current_ids_propositions

            if not is_possible:
                continue

            # Get the proposition that following this edge grants
            id_granted_proposition = list(mag.successors(id_edge))[0]

            # Add the proposition if it isn't already granted
            new_ids_propositions = current_ids_propositions.copy()
            if id_granted_proposition not in current_ids_propositions:
                bisect.insort(new_ids_propositions, id_granted_proposition)

            # Search if a node with such true propositions already exists
            similar_nodes = [
                i
                for i, ids_propositions in self.nodes(data="ids_propositions")
                if ids_propositions == new_ids_propositions
            ]
            if similar_nodes:
                # Search if there is already an edge between the source node
                # and the reached one
                similar_edges = [(src, dst) for src, dst in self.edges()
                                 if src == node[0] and dst == similar_nodes[0]]
                if not similar_edges and node[0] != similar_nodes[0]:
                    # Just add the edge
                    self.add_edge(node[0],
                                  similar_nodes[0],
                                  id_exploit=id_edge)
            else:
                # Create a brand new node
                new_node = (self.number_of_nodes(), {
                    "ids_propositions": new_ids_propositions
                })
                self.add_nodes_from([new_node])

                # Add an edge
                self.add_edge(node[0], new_node[0], id_exploit=id_edge)

                # Call recursively this function with the new node and and with
                # the used edge removed
                new_ids_edges = ids_edges.copy()
                new_ids_edges.remove(id_edge)
                self._fill_graph_recursively(mag, new_node, new_ids_edges)

    def _load_other_elements_from_json(self, data: dict):
        self.propositions = data["propositions"]
        self.exploits = data["exploits"]
        self.initial_node = data["initial_node"]
        self.final_node = data["final_node"]

    def _write_other_elements_in_data(self, data: dict) -> dict:
        new_data = data.copy()
        new_data["propositions"] = self.propositions
        new_data["exploits"] = self.exploits
        new_data["initial_node"] = self.initial_node
        new_data["final_node"] = self.final_node
        return new_data
