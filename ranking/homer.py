import numpy as np
from attack_graph import DependencyAttackGraph
from ranking.ranking import RankingMethod
from typing import Dict, List, Set, Tuple


class RiskQuantifier(RankingMethod):
    def __init__(self, graph: DependencyAttackGraph):
        super().__init__(list(graph.exploits))
        self.initial_graph = graph
        self.graph = self._set_up_graph(graph)
        self.exploit_probabilities = self._get_exploit_nodes_probabilities()

    def apply(self) -> Dict[int, float]:
        # Create the necessary arrays
        self.evaluated_nodes: Set[int] = set()
        self.dict_phi: Dict[int, float] = dict([(node, 0)
                                                for node in self.graph.nodes])
        self.dict_chi: Dict[int,
                            Set[int]] = dict([(node, set())
                                              for node in self.graph.nodes])
        self.dict_delta: Dict[int,
                              Set[int]] = dict([(node, set())
                                                for node in self.graph.nodes])

        # Create arrays useful to not compute the same value again
        self.list_stored_psi: List[Tuple[Dict[int, bool], Dict[int, bool],
                                         float]] = []
        self.list_stored_phi: List[Tuple[Dict[int, bool], float]] = []

        # Get the list of branch nodes
        self.branch_nodes = self._get_branch_nodes()

        # Treat the case of the root node
        self.evaluated_nodes.add(self.id_root_node)
        self.dict_phi[self.id_root_node] = 1

        # Main loop
        while len(self.evaluated_nodes) < self.graph.number_of_nodes():
            # Get a node that is ready for evaluation and its predecessors
            node, predecessors = self._get_node_ready_for_evaluation()

            if "id_proposition" in self.graph.nodes[node]:
                # Update the various arrays for this proposition node
                self.dict_phi[node] = 1 - self._evaluate_probability(
                    dict([(p, False) for p in predecessors]))

                for predecessor in predecessors:
                    self.dict_chi[node] |= self.dict_chi[predecessor]
                    self.dict_delta[node] &= self.dict_delta[predecessor]

                # If this node is a branch node then psi(n, n) = 1
                if node in self.branch_nodes:
                    self.list_stored_psi.append(({
                        node: True
                    }, {
                        node: True
                    }, 1))
            else:
                # Update the various arrays for this exploit node
                self.dict_phi[node] = self.exploit_probabilities[
                    node] * self._evaluate_probability(
                        dict([(p, True) for p in predecessors]))

                self.dict_chi[node] = self.branch_nodes & predecessors
                self.dict_delta[node] = self.branch_nodes & predecessors

                for predecessor in predecessors:
                    self.dict_chi[node] |= self.dict_chi[predecessor]
                    self.dict_delta[node] |= self.dict_delta[predecessor]

            # Add the node to the set of evaluated nodes
            self.evaluated_nodes.add(node)

        # Repopulate the graph with the nodes that have been removed
        risks = dict([(node, phi) for node, phi in self.dict_phi.items()
                      if node != self.id_root_node])
        for node in self.nodes_to_remove:
            risks[node] = 1

        return risks

    def get_score(self) -> float:
        risks = self.apply()
        score = sum(list(risks.values()))
        return score

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        ids_exploits_to_keep = self.ids_exploits.copy()
        ids_exploits_to_keep.remove(id_exploit)

        pruned_graph = self.initial_graph.get_pruned_graph(
            ids_exploits_to_keep)

        score = RiskQuantifier(pruned_graph).get_score()
        return score

    def _set_up_graph(self,
                      graph: DependencyAttackGraph) -> DependencyAttackGraph:
        new_graph = graph.copy()

        # Remove the proposition nodes that are initially true
        ids_initial_propositions = [
            id for id, data in new_graph.propositions.items()
            if data["initial"]
        ]
        self.nodes_to_remove = [
            node
            for node, id_proposition in new_graph.nodes(data="id_proposition")
            if id_proposition in ids_initial_propositions
        ]
        new_graph.remove_nodes_from(self.nodes_to_remove)

        # Create a root node
        self.id_root_node = int(
            np.setdiff1d(np.arange(new_graph.number_of_nodes() + 1),
                         list(new_graph.nodes))[0])
        new_graph.add_node(self.id_root_node, id_proposition=-1)

        # Add an edge from the root node to the exploits that are now without
        # any predecessor
        exploit_nodes_to_be_linked = [
            node for node, id_exploit in new_graph.nodes(data="id_exploit")
            if id_exploit is not None
            and list(new_graph.predecessors(node)) == []
        ]
        new_graph.add_edges_from([(self.id_root_node, node)
                                  for node in exploit_nodes_to_be_linked])

        return new_graph

    def _get_exploit_nodes_probabilities(self) -> Dict[int, float]:
        result = {}
        for node, id_exploit in self.graph.nodes(data="id_exploit"):
            if id_exploit is not None:
                # The probability is equal to the CVSS score divided by 10 (to
                # get a value between 0 and 1)
                probability = self.graph.exploits[id_exploit]["cvss"] / 10
                result[node] = probability
        return result

    def _get_branch_nodes(self) -> Set[int]:
        return set([
            node for node in self.graph.nodes
            if len(list(self.graph.successors(node))) > 1
        ])

    def _get_node_ready_for_evaluation(self) -> Tuple[int, Set[int]]:
        for node in self.graph.nodes:
            # Check that the node hasn't already been evaluated
            if node in self.evaluated_nodes:
                continue

            # Check that the predecessors of the node have all been evaluated
            predecessors = set(self.graph.predecessors(node))
            if self.evaluated_nodes >= predecessors:
                return node, predecessors

    def _evaluate_probability(self, node_polarities: Dict[int, bool]) -> float:
        nodes = set(node_polarities)

        # Check if this probability has already been computed
        for stored_node_polarities, value in self.list_stored_phi:
            if stored_node_polarities == node_polarities:
                return value

        # Find a d-separating set D
        D = set()
        for n in nodes:
            for m in nodes:
                if n != m:
                    D |= self.dict_chi[n] & self.dict_chi[m]

        if len(D) == 0:
            # There is no d-separating set so nodes are independant
            value = 1
            for node in nodes:
                phi = self.dict_phi[node]
                value *= phi if node_polarities[node] else 1 - phi
        else:
            # Compute the conditional probabilities given D
            value = 0

            # Create a list of lists to enumerate all the possible polarities
            # the nodes in D can take
            possible_polarities = [[True], [False]]
            for _ in range(len(D) - 1):
                possible_polarities = [
                    previous + [polarity] for previous in possible_polarities
                    for polarity in [True, False]
                ]

            # Compute the value by studying each possible configuration of
            # values for D
            for set_polarities in possible_polarities:
                D_polarities = dict([list(D)[i], set_polarities[i]]
                                    for i in range(len(D)))
                value += self._evaluate_conditional_probability(
                    node_polarities,
                    D_polarities) * self._evaluate_probability(D_polarities)

        # Save the probability for an eventual later use
        self.list_stored_phi.append((node_polarities, value))
        return value

    def _evaluate_conditional_probability(
            self, node_polarities: Dict[int, bool],
            D_polarities: Dict[int, bool]) -> float:
        # Check if this probability has already been computed
        for stored_psi in self.list_stored_psi:
            if stored_psi[0] == node_polarities and stored_psi[
                    1] == D_polarities:
                return stored_psi[2]

        if len(node_polarities) > 1:
            value = 1
            for node, polarity in node_polarities.items():
                value *= self._evaluate_conditional_probability(
                    {node: polarity}, D_polarities)
        else:
            node = list(node_polarities)[0]
            polarity = node_polarities[node]

            value = self._evaluate_single_node_conditional_probability(
                node, polarity, D_polarities)

        # Save the probability for an eventual later use
        self.list_stored_psi.append((node_polarities, D_polarities, value))
        return value

    def _evaluate_single_node_conditional_probability(
            self, node: int, polarity: bool,
            D_polarities: Dict[int, bool]) -> float:
        if polarity:
            # There is exactly one positive element
            J = set([d for d, pol in D_polarities.items() if pol])
            K = set([d for d, pol in D_polarities.items() if not pol])

            if node in J:
                return 1

            if node in K or len(K & self.dict_delta[node]) > 0:
                # The node or one of its denominator is negated in D
                return 0

            if len(set(D_polarities) & self.dict_chi[node]) == 0:
                # Set D does not affect the value of the node
                return self.dict_phi[node]

            predecessors = self.graph.predecessors(node)
            if "id_proposition" in self.graph.nodes[node]:
                return 1 - self._evaluate_conditional_probability(
                    dict([(p, False) for p in predecessors]), D_polarities)
            else:
                return self.exploit_probabilities[
                    node] * self._evaluate_conditional_probability(
                        dict([(p, True) for p in predecessors]), D_polarities)
        else:
            # There is exactly one negative element
            return 1 - self._evaluate_conditional_probability({node: True},
                                                              D_polarities)
