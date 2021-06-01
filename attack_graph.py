import json
import networkx as nx
import utils
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.sparse.coo import coo_matrix
from typing import Dict, List, Set


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
        #   - required_propositions: List[int],
        #   - cve_id: str
        #   - cvss: float
        self.exploits: Dict[int, dict] = {}

        # The id of the proposition that the attacker wants to reach
        self.goal_proposition: int = None

        # The ids of the nodes in which the final proposition is true
        self.goal_nodes: List[int] = []

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

                # If there is no successor to this proposition node, it means
                # that this proposition if the goal proposition
                self.goal_proposition = node

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
        self.propositions = dict([
            (int(id_proposition), proposition)
            for id_proposition, proposition in data["propositions"].items()
        ])
        self.exploits = dict([
            (int(id_exploit), exploit)
            for id_exploit, exploit in data["exploits"].items()
        ])
        self.goal_proposition = data["goal_proposition"]
        self.goal_nodes = data["goal_nodes"]

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
        data["goal_proposition"] = self.goal_proposition
        data["goal_nodes"] = self.goal_nodes

    def get_node_ordering(self) -> Dict[int, int]:
        ids_nodes = list(self.nodes)
        return dict([(id, i) for i, id in enumerate(ids_nodes)])

    def compute_adjacency_matrix(self, directed: bool = True) -> coo_matrix:
        if directed:
            network = self
        else:
            network = self.to_undirected()
        return nx.adjacency_matrix(network)

    def get_pruned_graph(self, ids_exploits: List[int]):
        return


class DependencyAttackGraph(BaseGraph):
    def fill_graph(self):
        i_node = 0

        # Add the nodes that correspond to the propositions
        for id in self.propositions:
            self.add_node(i_node, id_proposition=id)

            # If the id corresponds to the goal proposition, this node is the
            # goal node, which is unique
            if id == self.goal_proposition:
                self.goal_nodes.append(i_node)

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

        # Remove the nodes from which one can not reach the goal node
        self._remove_useless_nodes()

    def get_pruned_graph(self, ids_exploits: List[int]):
        # Copy this attack graph
        new_graph = self.copy()

        # Remove the nodes corresponding to the exploits to remove
        nodes_to_remove = []
        for node, id_exploit in new_graph.nodes(data="id_exploit"):
            if id_exploit is not None and id_exploit not in ids_exploits:
                nodes_to_remove.append(node)
        new_graph.remove_nodes_from(nodes_to_remove)

        # Remove the nodes that are now useless i.e. from which we can't reach
        # the goal node
        new_graph._remove_useless_nodes()

        return new_graph

    def _get_node_with_id_proposition(self, id_proposition: int) -> int:
        return [
            node
            for node, node_id_proposition in self.nodes(data="id_proposition")
            if node_id_proposition == id_proposition
        ][0]

    def _remove_useless_nodes(self):
        goal_node = self.goal_nodes[0]
        has_removed_nodes = True
        while has_removed_nodes:
            nodes_to_remove = []
            for node, data in self.nodes(data=True):
                # Check whether the node is a proposition or an exploit
                if "id_proposition" in data:
                    proposition = self.propositions[data["id_proposition"]]

                    # If the node has no successor and is not the goal node,
                    # we remove it
                    if node != goal_node and len(list(
                            self.successors(node))) == 0:
                        nodes_to_remove.append(node)

                    # If the node has no predecessor and does not correspond to
                    # a leaf proposition, we remove it
                    elif not proposition["initial"] and len(
                            list(self.predecessors(node))) == 0:
                        nodes_to_remove.append(node)

                        # If this node is the goal node, we remove it from the
                        # list of goal nodes
                        if node in self.goal_nodes:
                            self.goal_nodes.remove(node)
                else:
                    exploit = self.exploits[data["id_exploit"]]

                    # If the node has no successor, we remove it
                    if len(list(self.successors(node))) == 0:
                        nodes_to_remove.append(node)

                    # If the number of predecessors does not correspond to the
                    # number of required propositions, we remove it
                    elif len(list(self.predecessors(node))) != len(
                            exploit["required_propositions"]):
                        nodes_to_remove.append(node)

            self.remove_nodes_from(nodes_to_remove)
            has_removed_nodes = len(nodes_to_remove) > 0

    def copy(self, as_view=False):
        graph = super().copy(as_view=as_view)

        new_graph = DependencyAttackGraph()
        new_graph.load_nodes_and_edges(graph)
        new_graph.propositions = self.propositions.copy()
        new_graph.exploits = self.exploits.copy()
        new_graph.goal_proposition = self.goal_proposition
        new_graph.goal_nodes = self.goal_nodes.copy()

        return new_graph

    def _write_other_elements_in_data(self, data: dict):
        super()._write_other_elements_in_data(data)
        data["type"] = "dependency"


class StateAttackGraph(BaseGraph):
    def fill_graph(self):
        # Get the list of propositions that are true at the beginning
        ids_propositions = [
            i for i, data in self.propositions.items() if data["initial"]
        ]

        # Create and add the initial node
        self.add_node(0, ids_propositions=ids_propositions)

        # Fill the rest of the graph recursively
        self._fill_graph_recursively_from_node(0)

        # Sort the integer arrays in the graph
        for node, ids_propositions in self.nodes(data="ids_propositions"):
            self.nodes[node]["ids_propositions"] = sorted(ids_propositions)

        for src, dst, ids_exploits in self.edges(data="ids_exploits"):
            self.edges[src, dst]["ids_exploits"] = sorted(ids_exploits)

    def _fill_graph_recursively_from_node(self, node: int):
        current_ids_propositions: Set[int] = set(
            self.nodes[node]["ids_propositions"])

        if self.goal_proposition in current_ids_propositions:
            self.goal_nodes.append(node)
            return

        # Look for all the exploits that are possible and that grant a
        # proposition that is not already true
        ids_exploits_possible: Set[int] = set()
        for id_exploit, data in self.exploits.items():
            # Check that the proposition granted by the exploit is not already
            # true
            if data["granted_proposition"] in current_ids_propositions:
                continue

            required_ids_propositions: Set[int] = set(
                data["required_propositions"])

            # Check that the exploit can be performed
            if required_ids_propositions <= current_ids_propositions:
                ids_exploits_possible.add(id_exploit)

        for id_exploit in ids_exploits_possible:
            data = self.exploits[id_exploit]

            # Insert the granted proposition in the new propositions
            new_ids_propositions = current_ids_propositions.copy()
            new_ids_propositions.add(data["granted_proposition"])

            # Find if there are already nodes with such propositions
            similar_nodes = [
                i
                for i, ids_propositions in self.nodes(data="ids_propositions")
                if set(ids_propositions) == new_ids_propositions
            ]
            if len(similar_nodes) > 0:
                # Search if there is already an edge between the source node
                # and the reached one
                similar_edges = [(src, dst, edge_ids_exploits)
                                 for src, dst, edge_ids_exploits in self.edges(
                                     data="ids_exploits")
                                 if src == node and dst == similar_nodes[0]]
                if len(similar_edges) == 0:
                    self.add_edge(node,
                                  similar_nodes[0],
                                  ids_exploits=[id_exploit])
                else:
                    ids_exploits = similar_edges[0][2].copy()
                    ids_exploits.append(id_exploit)
                    self.edges[
                        similar_edges[0][0],
                        similar_edges[0][1]]["ids_exploits"] = ids_exploits

            else:
                # Add a new node
                new_node = self.number_of_nodes()
                self.add_node(
                    new_node,
                    ids_propositions=[int(i) for i in new_ids_propositions])

                # Add a new edge
                self.add_edge(node, new_node, ids_exploits=[id_exploit])

                # Call this function starting from the new node
                self._fill_graph_recursively_from_node(new_node)

    def copy(self, as_view=False):
        graph = super().copy(as_view=as_view)

        new_graph = StateAttackGraph()
        new_graph.load_nodes_and_edges(graph)
        new_graph.propositions = self.propositions.copy()
        new_graph.exploits = self.exploits.copy()
        new_graph.goal_proposition = self.goal_proposition
        new_graph.goal_nodes = self.goal_nodes.copy()

        return new_graph

    def get_pruned_graph(self, ids_exploits_to_keep: List[int]):
        # Copy this attack graph
        new_graph = self.copy()

        # Remove the edges corresponding to the exploits to remove
        edges_to_remove = []
        for src, dst, ids_exploits in new_graph.edges(data="ids_exploits"):
            new_ids_exploits = set(ids_exploits) & set(ids_exploits_to_keep)
            if len(new_ids_exploits) == 0:
                # Remove the edge totally
                edges_to_remove.append((src, dst))
            else:
                # Only remove the ids exploits from the list in the edge
                new_graph.edges[src,
                                dst]["ids_exploits"] = list(new_ids_exploits)
        new_graph.remove_edges_from(edges_to_remove)

        has_removed_nodes = True
        while has_removed_nodes:
            nodes_to_remove = []
            for node in new_graph.nodes:
                # The nodes that have no predecessors and aren't the initial
                # node must be removed
                if node != 0 and len(list(new_graph.predecessors(node))) == 0:
                    nodes_to_remove.append(node)

                    # If the node is a goal node, remove it from the list of
                    # goal nodes
                    if node in self.goal_nodes:
                        new_graph.goal_nodes.remove(node)

                # The nodes that have no successors and aren't one of the goal
                # nodes must be removed
                if node not in new_graph.goal_nodes and len(
                        list(new_graph.successors(node))) == 0:
                    nodes_to_remove.append(node)

            new_graph.remove_nodes_from(nodes_to_remove)
            has_removed_nodes = len(nodes_to_remove) > 0

        return new_graph

    def _write_other_elements_in_data(self, data: dict):
        super()._write_other_elements_in_data(data)
        data["type"] = "state"
