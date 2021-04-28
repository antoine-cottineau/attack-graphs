import numpy as np

from attack_graph import AttackGraph


class ValueIteration:
    def __init__(self, ag: AttackGraph):
        self.ag = ag

    def run(self, precision: float = 1e-4, lamb: float = 0.9):
        values = np.zeros(self.ag.number_of_nodes())
        chosen_successors = np.zeros(self.ag.number_of_nodes(), dtype=int)
        node_mapping = self.ag.get_node_mapping()
        delta = np.inf

        while delta > precision:
            new_values = np.zeros(self.ag.number_of_nodes())
            for node in self.ag.nodes:
                node_value = values[node_mapping[node]]
                reward = self.get_reward(node)
                successors = self.get_successors(node)

                # If the node is the final node, its value is always 1
                if len(successors) == 0:
                    new_values[node_mapping[node]] = 1
                    chosen_successors[node_mapping[node]] = node
                    continue

                # Find the best action
                best_value = -np.inf
                best_successor = None
                for successor in successors:
                    successor_value = values[node_mapping[successor[0]]]
                    probability = successor[1]

                    # The attacker either manages to perform the attack (with
                    # the above probability) or fails to. In the latter case,
                    # the attacker stays at the same node.
                    new_value = reward + lamb * (
                        probability * successor_value +
                        (1 - probability) * node_value)

                    if new_value > best_value:
                        best_value = new_value
                        best_successor = successor[0]

                # Update the current values and chosen actions
                new_values[node_mapping[node]] = best_value
                chosen_successors[node_mapping[node]] = best_successor

            # Compute delta
            delta = ValueIteration.compute_delta(values, new_values)

            values = new_values

        return values, chosen_successors

    def get_successors(self, node: int) -> list:
        successors = self.ag.successors(node)

        # In this simple case, we suppose that there is a same probability 0.9
        # to perform an exploit. In the future, the probability of an atomic
        # exploit should be calculated based on the CVSS score of the exploit.
        return [(successor, 0.9) for successor in successors]

    def get_reward(self, node: int) -> float:
        if node == self.ag.final_node:
            return 1
        else:
            return 0

    @staticmethod
    def compute_delta(before: np.ndarray, after: np.ndarray) -> float:
        return np.linalg.norm(before - after, ord=2)
