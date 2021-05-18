import json
import networkx as nx
import numpy as np
import utils
import xml.etree.ElementTree as ET
from bisect import insort
from pathlib import Path
from scipy.sparse.coo import coo_matrix
from typing import Dict, List


class BaseGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()

        # Elements in a proposition:
        #   - id_proposition: int (key of the dictionary)
        #   - text: str
        #   - initial: bool
        self.propositions: Dict[int, dict] = {}

        # Elements in an exploit:
        #   - id_exploit: int (key of the dictionary)
        #   - text: str
        #   - granted_proposition: int
        #   - required_propositions: List[int]
        self.exploits: Dict[int, dict] = {}

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
        graph = self._create_mulval_graph(path, string)

        # Fill the exploits dictionary by using BFS
        visited_propositions = []
        nodes_to_visit_next = []

        # Add the root nodes to nodes_to_visit_next
        for node in graph.nodes():
            predecessors = list(graph.predecessors(node))
            if predecessors == []:
                nodes_to_visit_next.append(node)

        # Apply BFS
        while nodes_to_visit_next != []:
            node = nodes_to_visit_next[0]
            data = graph.nodes[node]

            if data["type"] == "proposition":
                # The node is a proposition
                visited_propositions.append(node)
                successors = list(graph.successors(node))

                # Add the successors to the list of nodes to visit
                for successor in successors:
                    if successor not in nodes_to_visit_next:
                        nodes_to_visit_next.append(successor)

                # Remove the node to the list of nodes to visit
                nodes_to_visit_next = nodes_to_visit_next[1:]
            elif data["type"] == "exploit":
                # The node is an exploit
                predecessors = list(graph.predecessors(node))
                successors = list(graph.successors(node))

                granted_proposition = successors[0]

                # Compute the required propositions
                required_propositions = sorted(predecessors)
                is_possible = [
                    i for i in required_propositions
                    if i not in visited_propositions
                ] == []

                # Remove the node from the list of nodes to visit
                nodes_to_visit_next = nodes_to_visit_next[1:]
                if is_possible:
                    # Add the exploit
                    exploit = dict(text=data["text"],
                                   granted_proposition=granted_proposition,
                                   required_propositions=required_propositions)
                    self.exploits[node] = exploit

                    # Add the granted proposition at the beginning of the list
                    # of nodes to visit
                    nodes_to_visit_next.insert(0, successors[0])
                else:
                    # Add the node at the end of the list of nodes to visit
                    nodes_to_visit_next.append(node)

        # Fill the graph by using the computed propositions and exploits
        self.fill_graph()

    def _create_mulval_graph(self,
                             path: str = None,
                             string: str = None) -> nx.DiGraph:
        if path is None and string is None:
            raise Exception("Path or string must be not None")

        # Parse the file or the string
        if path is None:
            tree = ET.ElementTree(ET.fromstring(string))
        else:
            tree = ET.parse(path)

        root = tree.getroot()
        nodes = root.findall(path="vertices/vertex")
        edges = root.findall(path="arcs/arc")

        # Start by creating an instance of Networkx with the nodes and edges of
        # the file
        graph = nx.DiGraph()

        # Add the nodes to the graph
        for node in nodes:
            id_proposition = int(node.find("id").text)
            text = node.find("fact").text
            type = node.find("type").text

            if type == "OR" or type == "LEAF":
                # This node corresponds to a proposition node
                graph.add_node(id_proposition, text=text, type="proposition")
                self.propositions[id_proposition] = dict(
                    text=text, initial=type == "LEAF")
            elif type == "AND":
                # This node corresponds to an exploit node
                graph.add_node(id_proposition, text=text, type="exploit")

        # Add the edges to the graph
        for edge in edges:
            src = int(edge.find("src").text)
            dst = int(edge.find("dst").text)

            # Source and destination nodes seem inverted in the attack graph
            # files generated by MulVAL
            graph.add_edge(dst, src)

        return graph

    def fill_graph(self):
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
        self.propositions = data["propositions"]
        self.exploits = data["exploits"]

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
        self._write_other_elements_in_data(data)
        return data

    def _write_other_elements_in_data(self, data: dict):
        data["propositions"] = self.propositions
        data["exploits"] = self.exploits

    def get_node_ordering(self) -> dict:
        ids_nodes = list(self.nodes)
        return dict([(id, i) for i, id in enumerate(ids_nodes)])

    def compute_adjacency_matrix(self, directed: bool = True) -> coo_matrix:
        if directed:
            network = self
        else:
            network = self.to_undirected()
        return nx.adjacency_matrix(network)


class DependencyAttackGraph(BaseGraph):
    def fill_graph(self):
        i_node = 0

        # Add the nodes that correspond to the propositions
        for id in self.propositions:
            self.add_node(i_node, id_proposition=id)
            i_node += 1

        # Add the nodes that correspond to the exploits and the edges that link
        # them to the propositions
        for id, data in self.exploits.items():
            self.add_node(i_node, id_exploit=id)

            # Add the edge to the granted proposition
            node_granted_proposition = self._get_node_with_id_proposition(
                data["granted_proposition"])
            self.add_edge(i_node, node_granted_proposition)

            # Add the edges from the required propositions
            for id_proposition in data["required_propositions"]:
                node_required_proposition = self._get_node_with_id_proposition(
                    id_proposition)
                self.add_edge(node_required_proposition, i_node)

            i_node += 1

    def _get_node_with_id_proposition(self, id_proposition: int) -> int:
        return [
            node
            for node, node_id_proposition in self.nodes(data="id_proposition")
            if node_id_proposition == id_proposition
        ][0]

    def copy(self, as_view=False):
        graph = super().copy(as_view=as_view)

        new_graph = DependencyAttackGraph()
        new_graph.load_nodes_and_edges(graph)
        new_graph.propositions = self.propositions.copy()
        new_graph.exploits = self.exploits.copy()

        return new_graph


class StateAttackGraph(BaseGraph):
    def fill_graph(self):
        # Get the list of propositions that are true at the beginning
        ids_propositions = [
            i for i, data in self.propositions.items() if data["initial"]
        ]

        # Create and add the initial node
        self.add_node(0, ids_propositions=ids_propositions)

        # Fill the rest of the graph recursively
        self.fill_graph_recursively_from_node(0)

    def fill_graph_recursively_from_node(self, node: int):
        current_ids_propositions: List[int] = self.nodes[node][
            "ids_propositions"]

        # Look for all the exploits that are possible and that grant a
        # proposition that is not already true
        ids_exploits_possible: List[int] = []
        for id_exploit, data in self.exploits.items():
            # Check that the proposition granted by the exploit is not already
            # true
            if data["granted_proposition"] in current_ids_propositions:
                continue

            # Check that the exploit can be performed
            common_propositions = np.intersect1d(data["required_propositions"],
                                                 current_ids_propositions)
            if len(data["required_propositions"]) == len(common_propositions):
                # The exploit is possible
                ids_exploits_possible.append(id_exploit)

        # If no exploit is possible, this node is the final node
        if ids_exploits_possible == []:
            self.final_node = node

        for id_exploit in ids_exploits_possible:
            data = self.exploits[id_exploit]

            # Insert the granted proposition in the new propositions
            new_ids_propositions = current_ids_propositions.copy()
            insort(new_ids_propositions, data["granted_proposition"])

            # Find if there are already nodes with such propositions
            similar_nodes = [
                i
                for i, ids_propositions in self.nodes(data="ids_propositions")
                if ids_propositions == new_ids_propositions
            ]
            if similar_nodes != []:
                # Search if there is already an edge between the source node
                # and the reached one
                similar_edges = [(src, dst) for src, dst in self.edges
                                 if src == node and dst == similar_nodes[0]]
                if similar_edges == []:
                    # Just add the edge
                    self.add_edge(node,
                                  similar_nodes[0],
                                  id_exploit=id_exploit)
            else:
                # Add a new node
                new_node = self.number_of_nodes()
                self.add_node(
                    new_node,
                    ids_propositions=[int(i) for i in new_ids_propositions])

                # Add a new edge
                self.add_edge(node, new_node, id_exploit=id_exploit)

                # Call this function starting from the new node
                self.fill_graph_recursively_from_node(new_node)

    def copy(self, as_view=False):
        graph = super().copy(as_view=as_view)

        new_graph = StateAttackGraph()
        new_graph.load_nodes_and_edges(graph)
        new_graph.propositions = self.propositions.copy()
        new_graph.exploits = self.exploits.copy()
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
            for i in new_graph.nodes:
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

    def _load_other_elements_from_json(self, data: dict):
        self.final_node = data["final_node"]

    def _write_other_elements_in_data(self, data: dict):
        data["final_node"] = self.final_node
