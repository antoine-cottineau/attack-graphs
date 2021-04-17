import bisect
import networkx as nx
import numpy as np

from attack_graph import AttackGraph


class Generator:
    def __init__(self,
                 n_propositions: int = 20,
                 n_initial_propositions: int = 10,
                 n_exploits: int = 20):
        self.n_propositions = n_propositions
        self.n_initial_propositions = n_initial_propositions
        self.n_exploits = n_exploits

    def generate(self):
        exploits = self.generate_exploits()

        # Create a graph
        graph = nx.DiGraph()

        # Create an initial node and add it to the graph
        initial_node = (0, {
            "ids_propositions":
            list(range(self.n_initial_propositions))
        })
        graph.add_nodes_from([initial_node])

        # Add the other nodes recursively
        self.add_nodes(initial_node, exploits, graph)

        # Create the attack graph
        ag = AttackGraph()
        ag.add_nodes_from(graph.nodes(data=True))
        ag.add_edges_from(graph.edges(data=True))

        # Add other variables to the attack graph
        ag.propositions = {}
        for i in range(self.n_propositions):
            ag.propositions[i] = (i, "Randomly generated")
        ag.create_proposition_mapping()

        return ag

    def generate_exploits(self):
        # Get the list of the propositions that need to be granted by at least
        # one exploit
        propositions_to_grant = np.arange(self.n_initial_propositions,
                                          self.n_propositions)

        # We want at least one exploit that grants each one of the propositions
        # that are not true at the beginning
        granted_propositions = np.copy(propositions_to_grant)

        # Sample with replacement from propositions_to_grant to get n_exploits
        # propositions to grant in total
        if granted_propositions.shape[0] < self.n_exploits:
            new_granted_propositions = np.random.choice(
                propositions_to_grant,
                self.n_exploits - granted_propositions.shape[0])

            granted_propositions = np.concatenate(
                [granted_propositions, new_granted_propositions])

            granted_propositions = np.sort(granted_propositions)

        # An exploit is defined by the propositions that must be true to
        # perform it and by the proposition it grants
        exploits = []

        # Each exploit is given an id and a small description
        id_exploit = 0

        # Create the exploits by sampling some propositions that must be true
        # so that an exploit can be performed
        for granted_proposition in granted_propositions:
            already_exists = True
            while already_exists:
                # Sample a number between 1 and 4 that corresponds to how many
                # propositions must be true so that the exploit can be
                # performed
                n_required_propositions = np.random.randint(4, 6)

                # Sample the necessary propositions
                probabilities = np.exp(np.arange(granted_proposition))
                probabilities /= np.sum(probabilities)
                required_propositions = np.random.choice(
                    granted_proposition,
                    n_required_propositions,
                    replace=False)
                required_propositions = np.sort(required_propositions)

                # Create the exploit and check that it doesn't already exist
                exploit = (granted_proposition, list(required_propositions),
                           id_exploit)
                already_exists = exploit in exploits

            # Add the new exploit and update the id
            exploits.append(exploit)
            id_exploit += 1

        return exploits

    def add_nodes(self, node: tuple, exploits: list, graph: nx.DiGraph):
        current_ids_propositions = node[1]["ids_propositions"]
        # Look for all the exploits that are possible and that grant a
        # proposition that is not already true
        possible_exploits = []
        for exploit in exploits:
            # Check that the proposition granted by the exploit is not already
            # true
            if exploit[0] in current_ids_propositions:
                continue

            # Check that the exploit is possible
            common_propositions = np.intersect1d(exploit[1],
                                                 current_ids_propositions)
            if len(exploit[1]) == common_propositions.shape[0]:
                # The exploit is possible
                possible_exploits.append(exploit)

        for exploit in possible_exploits:
            new_ids_propositions = current_ids_propositions.copy()
            bisect.insort(new_ids_propositions, exploit[0])

            # Find if there is already a node with such propositions
            similar_nodes = [
                i
                for i, ids_propositions in graph.nodes(data="ids_propositions")
                if ids_propositions == new_ids_propositions
            ]
            if similar_nodes:
                # Search if there is already an edge between the source node
                # and the reached one
                similar_edges = [(src, dst) for src, dst in graph.edges()
                                 if src == node[0] and dst == similar_nodes[0]]
                if not similar_edges and node[0] != similar_nodes[0]:
                    # Just add the edge
                    graph.add_edge(node[0],
                                   similar_nodes[0],
                                   id_exploit=exploit[2],
                                   exploit="Randomly generated ({})".format(
                                       exploit[2]))
            else:
                # Create a brand new node
                new_node = (graph.number_of_nodes(), {
                    "ids_propositions": new_ids_propositions
                })
                graph.add_nodes_from([new_node])

                # Add an edge
                graph.add_edge(node[0],
                               new_node[0],
                               id_exploit=exploit[2],
                               exploit="Randomly generated ({})".format(
                                   exploit[2]))

                # Call recursively this function with the new node and and with
                # the used edge removed
                self.add_nodes(new_node, exploits, graph)
