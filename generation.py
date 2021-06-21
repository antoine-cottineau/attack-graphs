import networkx as nx
import numpy as np
from attack_graph import DependencyAttackGraph, StateAttackGraph
from cve import ExploitFetcher
from typing import Dict, List, Tuple


class Generator:
    def __init__(self,
                 n_exploits: int = 30,
                 exploits_prob_n_predecessors: Dict[int, float] = {
                     1: 0.45,
                     2: 0.3,
                     3: 0.25
                 },
                 propositions_prob_n_successors: Dict[int, float] = {
                     1: 0.45,
                     2: 0.35,
                     3: 0.2
                 },
                 probability_new_proposition: float = 0.4):
        self.n_exploits = n_exploits
        self.exploits_prob_n_predecessors = exploits_prob_n_predecessors
        self.propositions_prob_n_successors = propositions_prob_n_successors
        self.probability_new_proposition = probability_new_proposition

        self.exploits: Dict[int, dict] = {}
        self.exploits_fake_data = ExploitFetcher().get_fake_exploit_list(
            n_exploits, shuffle=True)
        self.propositions: Dict[int, dict] = {}
        self.propositions_n_successors: Dict[int, int] = {}
        self.graph = nx.DiGraph()
        self.goal_proposition = None

    def generate_both_graphs(
            self) -> Tuple[StateAttackGraph, DependencyAttackGraph]:
        self._generate_exploits()

        state_graph = StateAttackGraph()
        state_graph.propositions = self.propositions.copy()
        state_graph.exploits = self.exploits.copy()
        state_graph.goal_proposition = self.goal_proposition
        state_graph.fill_graph()

        dependency_graph = DependencyAttackGraph()
        dependency_graph.propositions = self.propositions.copy()
        dependency_graph.exploits = self.exploits.copy()
        dependency_graph.goal_proposition = self.goal_proposition
        dependency_graph.fill_graph()

        return state_graph, dependency_graph

    def generate_dependency_attack_graph(self) -> DependencyAttackGraph:
        self._generate_exploits()

        dependency_graph = DependencyAttackGraph()
        dependency_graph.propositions = self.propositions.copy()
        dependency_graph.exploits = self.exploits.copy()
        dependency_graph.goal_proposition = self.goal_proposition
        dependency_graph.fill_graph()

        return dependency_graph

    def _generate_exploits(self):
        # Stop the recursion when we have enough exploits and only one goal
        # proposition
        if len(self.exploits) == self.n_exploits:
            self._finish_graph()
            return

        # Get the list of available propositions
        available_propositions = self._get_available_propositions()

        if len(available_propositions) >= self.n_exploits - len(self.exploits):
            # Stop creating new proposition and finish the graph
            self.probability_new_proposition = 0

        # Sample the number of propositions that are required for this exploit
        # to be performed
        n_required_propositions: int = np.random.choice(
            list(self.exploits_prob_n_predecessors),
            p=list(self.exploits_prob_n_predecessors.values()))

        # Add and/or link propositions to the exploit
        predecessors = []
        required_propositions = []
        for _ in range(n_required_propositions):
            new = np.random.rand() < self.probability_new_proposition
            if new or len(available_propositions) == 0:
                # Create a new proposition
                node, id_proposition = self._add_new_proposition(initial=True)
            else:
                # Sample one proposition from the list of available
                # propositions
                position = np.random.choice(len(available_propositions))
                node, id_proposition = available_propositions.pop(position)

            predecessors.append(node)
            required_propositions.append(id_proposition)

        # Create a new proposition representing the granted proposition
        successor, granted_proposition = self._add_new_proposition(
            initial=False)

        # Add the new exploit
        self._add_new_exploit(required_propositions, granted_proposition,
                              predecessors, successor)

        # Call the function recursively
        self._generate_exploits()

    def _finish_graph(self):
        # Look for the proposition nodes that do not have successors
        nodes_to_merge = []
        for node, id_proposition in self.graph.nodes(data="id_proposition"):
            if id_proposition is None:
                continue
            n_successors = len(list(self.graph.successors(node)))
            if n_successors == 0:
                nodes_to_merge.append((node, id_proposition))

        target_node, target_id_proposition = nodes_to_merge[0]
        self.goal_proposition = target_id_proposition

        if len(nodes_to_merge) == 1:
            return

        for i_node in range(1, len(nodes_to_merge)):
            node, id_proposition = nodes_to_merge[i_node]
            predecessors = list(self.graph.predecessors(node))

            for predecessor in predecessors:
                # Update the exploit
                id_exploit = self.graph.nodes[predecessor]["id_exploit"]
                self.exploits[id_exploit][
                    "granted_proposition"] = target_id_proposition

                # Update the graph
                self.graph.add_edge(predecessor, target_node)

            # Remove the node
            self.graph.remove_node(node)

    def _get_available_propositions(self) -> List[Tuple[int, int]]:
        propositions = []
        for node, id_proposition in self.graph.nodes(data="id_proposition"):
            if id_proposition is None:
                continue
            n_successors = len(list(self.graph.successors(node)))
            n_required_successors = self.propositions_n_successors[
                id_proposition]
            if n_successors < n_required_successors:
                propositions.append((node, id_proposition))
        return propositions

    def _add_new_proposition(self, initial: bool) -> Tuple[int, int]:
        node = self.graph.number_of_nodes()
        id_proposition = len(self.propositions)

        # Create and add the proposition
        proposition = dict(text="Randomly generated {}".format(id_proposition),
                           initial=initial)
        self.propositions[id_proposition] = proposition

        # Sample the number of successors for this proposition
        n_successors = np.random.choice(
            list(self.propositions_prob_n_successors),
            p=list(self.propositions_prob_n_successors.values()))
        self.propositions_n_successors[id_proposition] = n_successors

        # Update the graph
        self.graph.add_node(node, id_proposition=id_proposition)

        return node, id_proposition

    def _add_new_exploit(self, required_propositions: List[int],
                         granted_proposition: int, predecessors: List[int],
                         successor: List[int]):
        node = self.graph.number_of_nodes()
        id_exploit = len(self.exploits)

        # Create and add the exploit
        exploit_fake_data = self.exploits_fake_data[id_exploit]
        exploit = dict(text=exploit_fake_data["text"],
                       granted_proposition=granted_proposition,
                       required_propositions=required_propositions,
                       cve_id=exploit_fake_data["cve_id"],
                       cvss=exploit_fake_data["cvss"])
        self.exploits[id_exploit] = exploit

        # Update the graph
        self.graph.add_node(node, id_exploit=id_exploit)
        for predecessor in predecessors:
            self.graph.add_edge(predecessor, node)
        self.graph.add_edge(node, successor)
