import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod
from scipy.sparse import csr_matrix


class ProbabilisticPath(RankingMethod):
    def __init__(self, graph: StateAttackGraph):
        super().__init__(list(graph.exploits))
        self.graph = graph

    def apply(self) -> float:
        self._create_Q_and_R()
        self._create_N()
        self._create_B()

        # We only return the  sum of the probabilities from the initial node to
        # the goal nodes
        result = sum(self.B[0])
        return result

    def get_score(self) -> float:
        score = self.apply()
        return score

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        pruned_graph = self._get_pruned_graph(self.graph, id_exploit)

        if pruned_graph is None:
            return float("-inf")
        else:
            score = ProbabilisticPath(pruned_graph).get_score()
            return score

    def _create_Q_and_R(self):
        transient_nodes = set(self.graph.nodes) - set(self.graph.goal_nodes)
        absorbing_nodes = set(self.graph.goal_nodes)

        # Q is a sub matrix of P that contains transitions from transient
        # nodes to transient nodes
        Q = np.zeros((len(transient_nodes), len(transient_nodes)))

        # R is a sub matrix of P that contains transitions from transient
        # nodes to absorbing nodes
        R = np.zeros((len(transient_nodes), len(absorbing_nodes)))

        all_nodes = list(transient_nodes) + list(absorbing_nodes)
        node_ordering = dict([(all_nodes[i], i)
                              for i in range(len(all_nodes))])

        # Create Q and R
        for node in transient_nodes:
            i = node_ordering[node]

            # Compute the probability of each outgoing edge
            probabilities = dict([(s, self.graph.get_edge_probability(node, s))
                                  for s in self.graph.successors(node)])
            normalization_constant = sum(probabilities.values())

            # Add the normalized probability to Q or R
            for successor, probability in probabilities.items():
                j = node_ordering[successor]
                transition = probability / normalization_constant
                if successor in transient_nodes:
                    Q[i, j] = transition
                else:
                    R[i, j - len(transient_nodes)] = transition

        self.Q = csr_matrix(Q)
        self.R = csr_matrix(R)

    def _create_N(self):
        n_transient_nodes = self.Q.shape[0]
        N = csr_matrix(np.identity(n_transient_nodes))
        Q_power_i = csr_matrix(np.identity(n_transient_nodes))
        while Q_power_i.sum() > 1e-15:
            Q_power_i = Q_power_i.dot(self.Q)
            N += Q_power_i
        self.N = N

    def _create_B(self):
        B: csr_matrix = self.N.dot(self.R)
        B = B.toarray()
        self.B = B
