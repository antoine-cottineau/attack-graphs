import numpy as np
from pathlib import Path

from attack_graph import AttackGraph
from docker_handler import DockerHandler
from embedding import Embedding


class DeepWalk(Embedding):

    """
    Class that implements the DeepWalk algorithm introduced by Perozzi et al.

    :param AttackGraph ag: The attack graph.
    :param int dim_embedding: The dimension of the embedding of each node.
    :param str prefix: The prefix of the input file that should be
    created.
    """
    def __init__(self, ag: AttackGraph, dim_embedding: int, prefix: str):
        self.ag = ag
        self.dim_embedding = dim_embedding
        self.prefix = prefix

        self.dh = DockerHandler("deepwalk")

    def run(self):
        """
        Run the DeepWalk algorithm.
        """
        self.dh.run_container()

        self.create_adjacency_list()
        self.dh.transfer_folder("deepwalk_input", "experiments", self.prefix)

        self.run_deepwalk()

        self.create_embeddings()

    def create_adjacency_list(self):
        """
        Create the necessary input file for DeepWalk, an adjacency list.
        """
        Path("deepwalk_input").mkdir(exist_ok=True)

        with open("deepwalk_input/{}.adjlist".format(self.prefix), "w") as f:
            for state in self.ag.states:
                if state.id_ != 0:
                    f.write("\n")

                f.write("{} ".format(state.id_))

                # Write the neighbours
                neighbours = sorted([*state.in_] + [*state.out])
                f.write(" ".join([str(i) for i in neighbours]))

    def run_deepwalk(self):
        """
        Run the DeepWalk algorithm in the container
        """
        command = "deepwalk "

        # Precise the adjacency list
        command += "--input /experiments/deepwalk_input/{}.adjlist ".format(
            self.prefix)

        # Precise the dimension of the embedding
        command += "--representation-size {} ".format(self.dim_embedding)

        # Precise the output
        command += "--output /experiments/{}.embeddings".format(self.prefix)

        self.dh.run_command(command)

    def create_embeddings(self):
        """
        Create the embeddings from the output file.
        """
        # Create a folder for DeepWalk results
        Path("deepwalk_output").mkdir(exist_ok=True)

        # Get the embeddings file
        self.dh.copy_folder_from_container(
            "experiments", "deepwalk_output",
            ["{}.embeddings".format(self.prefix)])

        # Read the result file
        embeddings = np.zeros((self.ag.N, self.dim_embedding))
        with open("deepwalk_output/{}.embeddings".format(self.prefix)) as f:
            # The first line isn't useful, it is just the dimensions
            line = f.readline()

            while line:
                line = f.readline()
                if line == "":
                    continue
                elements = line.split(" ")
                embeddings[int(elements[0])] = [float(i) for i in elements[1:]]

        self.embeddings = embeddings
