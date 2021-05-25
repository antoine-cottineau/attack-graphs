import numpy as np
from attack_graph import DependencyAttackGraph, StateAttackGraph
from cve import ExploitFetcher
from typing import Dict


class Generator:
    def __init__(self,
                 n_propositions: int = 20,
                 n_initial_propositions: int = 10,
                 n_exploits: int = 20,
                 min_exploit_requirement: int = 1,
                 max_exploit_requirement: int = 4):
        self.n_propositions = n_propositions
        self.n_initial_propositions = n_initial_propositions
        self.n_exploits = n_exploits
        self.min_exploit_requirement = min_exploit_requirement
        self.max_exploit_requirement = max_exploit_requirement

    def generate_state_attack_graph(self) -> StateAttackGraph:
        # Create the attack graph
        graph = StateAttackGraph()

        graph.propositions = self.propositions.copy()
        graph.exploits = self.exploits.copy()

        # The goal proposition is the last one
        graph.goal_proposition = self.n_propositions - 1

        # Fill the graph
        graph.fill_graph()

        return graph

    def generate_dependency_attack_graph(self) -> DependencyAttackGraph:
        # Create the attack graph
        graph = DependencyAttackGraph()

        graph.propositions = self.propositions.copy()
        graph.exploits = self.exploits.copy()

        # The goal proposition is the last one
        graph.goal_proposition = self.n_propositions - 1

        # Fill the graph
        graph.fill_graph()

        return graph

    def generate_propositions_and_exploits(self):
        # Generate the propositions and the exploits
        self._generate_propositions()
        self._generate_exploits()

        # Remove the propositions that aren't used by any exploit
        self._remove_unused_propositions()

    def _generate_propositions(self):
        propositions: Dict[int, dict] = {}
        for id_proposition in range(self.n_propositions):
            proposition = dict(
                text="Randomly generated {}".format(id_proposition),
                initial=id_proposition < self.n_initial_propositions)
            propositions[id_proposition] = proposition
        self.propositions = propositions

    def _generate_exploits(self) -> Dict[int, dict]:
        # Get the list of the propositions that need to be granted by at least
        # one exploit
        propositions_to_grant = np.arange(self.n_initial_propositions,
                                          self.n_propositions)

        # We want at least one exploit to grant each one of the propositions
        # that are not true at the beginning
        granted_propositions = np.copy(propositions_to_grant)

        # Sample with replacement from propositions_to_grant to get n_exploits
        # propositions to grant in total
        if len(granted_propositions) < self.n_exploits:
            new_granted_propositions = np.random.choice(
                propositions_to_grant,
                size=self.n_exploits - granted_propositions.shape[0])

            granted_propositions = np.concatenate(
                [granted_propositions, new_granted_propositions])

            granted_propositions = np.sort(granted_propositions)

        # Create a dictionary of exploits
        exploits: Dict[int, dict] = {}

        # Get some real-world exploits
        fake_data = ExploitFetcher().get_fake_exploit_list(self.n_exploits)

        # Create the exploits by sampling some propositions that must be true
        # so that an exploit can be performed
        id_exploit = 0
        for granted_proposition in granted_propositions:
            already_exists = True
            while already_exists:
                # Sample a number between min_exploit_requirement and
                # max_exploit_requirement that corresponds to how many
                # propositions must be true so that the exploit can be
                # performed
                n_required_propositions = np.random.randint(
                    self.min_exploit_requirement,
                    self.max_exploit_requirement + 1)

                # Sample the necessary propositions
                probabilities = np.arange(granted_proposition, dtype=float)
                probabilities /= np.sum(probabilities)
                required_propositions = np.random.choice(
                    granted_proposition,
                    size=n_required_propositions,
                    replace=False,
                    p=probabilities)
                required_propositions = np.sort(required_propositions)

                # Create the exploit and check that it doesn't already exist
                exploit = dict(granted_proposition=int(granted_proposition),
                               required_propositions=[
                                   int(i) for i in required_propositions
                               ])
                already_exists = exploit in exploits.values()

            # Add data to the new exploit
            exploit["cve_id"] = fake_data[id_exploit]["cve_id"]
            exploit["text"] = fake_data[id_exploit]["text"]
            exploit["cvss"] = fake_data[id_exploit]["cvss"]

            # Add the new exploit and update the id
            exploits[id_exploit] = exploit
            id_exploit += 1

        self.exploits = exploits

    def _remove_unused_propositions(self):
        unused_propositions = np.array(list(self.propositions.keys()))
        for data in self.exploits.values():
            required_propositions = data["required_propositions"]
            unused_propositions = np.setdiff1d(unused_propositions,
                                               required_propositions,
                                               assume_unique=True)

            granted_proposition = data["granted_proposition"]
            unused_propositions = np.setdiff1d(unused_propositions,
                                               granted_proposition,
                                               assume_unique=True)

        for id_proposition in unused_propositions:
            del self.propositions[id_proposition]
