import numpy as np
from attack_graph import StateAttackGraph
from ranking.ranking import RankingMethod
from scipy.sparse import csr_matrix


class ProbabilisticPath(RankingMethod):
    def __init__(self, graph: StateAttackGraph):
        super().__init__(graph)

    def apply(self) -> float:
        self._create_Q_and_R()
        self._create_N()
        self._create_B()

        # We only return the expected path length from the initial state
        result = self.B[0].max()
        return result

    def _get_score(self) -> float:
        score = self.apply()
        return score

    def _get_score_for_graph(self, graph: StateAttackGraph) -> float:
        score = ProbabilisticPath(graph)._get_score()
        return score

    def _get_edge_probability(self, src: int, dst: int) -> float:
        ids_exploits = self.graph.edges[src, dst]["ids_exploits"]

        # The probability of the action being successful is equal to the
        # max of the probilities of the exploits
        probabilities = []
        for id_exploit in ids_exploits:
            # The probability is equal to the CVSS score divided by 10 (to
            # get a value between 0 and 1)
            probability = self.graph.exploits[id_exploit]["cvss"] / 10
            probabilities.append(probability)
        return max(probabilities)

    def _create_Q_and_R(self):
        transient_nodes = set(self.graph.nodes) - set(self.graph.goal_nodes)
        absorbing_states = set(self.graph.goal_nodes)

        # Q is a sub matrix of P that contains transitions from transient
        # nodes to transient nodes
        Q = np.zeros((len(transient_nodes), len(transient_nodes)))

        # R is a sub matrix of P that contains transitions from transient
        # nodes to absorbing nodes
        R = np.zeros((len(transient_nodes), len(absorbing_states)))

        all_nodes = list(transient_nodes) + list(absorbing_states)
        node_ordering = dict([(all_nodes[i], i)
                              for i in range(len(all_nodes))])

        # Create Q and R
        for node in transient_nodes:
            i = node_ordering[node]

            # Find the successors of the node
            successors = set(self.graph.successors(node))

            # Compute the probability of each outgoing edge
            probabilities = dict([(s, self._get_edge_probability(node, s))
                                  for s in successors])
            sum_probabilities = sum(probabilities.values())

            # Add the normalized probability to Q or R
            for successor, probability in probabilities.items():
                j = node_ordering[successor]
                transition = probability / sum_probabilities
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
