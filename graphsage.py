import json
import numpy as np
import docker
import tarfile
from pathlib import Path, PurePath
from attack_graph import AttackGraph
from embedding import Embedding


class FileCreator:
    """
    Class used to create the necessary input files for GraphSAGE.

    :param AttackGraph ag: The attack graph.
    :param str prefix: The prefix of each one of the input files that should be
    created.
    """
    def __init__(self, ag: AttackGraph, prefix: str):
        self.ag = ag
        self.prefix = prefix

    def create_files(self):
        """
        Create the input files.
        """
        Path("graphsage_input").mkdir(exist_ok=True)

        self.create_G()
        self.create_id_map()
        self.create_class_map()
        self.create_feats()
        self.create_walks()

    def create_G(self):
        """
        Create the file {prefix}-G.json which lists all the nodes and edges.
        """
        G = {
            "directed": False,
            "graph": {
                "name": "graph"
            },
            "nodes": [],
            "links": []
        }

        for state in self.ag.states:
            node = {
                "feature": [],
                "id": state.id_,
                "label": [1],
                "test": False,
                "val": False
            }
            G["nodes"].append(node)

            for id_out in state.out:
                link = {
                    "source": state.id_,
                    "target": id_out,
                    "test_removed": False,
                    "train_removed": False
                }
            G["links"].append(link)

        with open("graphsage_input/{}-G.json".format(self.prefix), "w") as f:
            json.dump(G, f, indent=2)

    def create_id_map(self):
        """
        Create the file {prefix}-id_map.json which links node ids with an
        integer.
        """
        id_map = {}

        for i in range(self.ag.N):
            state = self.ag.states[i]
            id_map[state.id_] = i

        with open("graphsage_input/{}-id_map.json".format(self.prefix),
                  "w") as f:
            json.dump(id_map, f, indent=2)

    def create_class_map(self):
        """
        Create the file {prefix}-class_map.json which links node with the
        classes they belong to. It is still necessary even when we don't aim
        to perform classification.
        """
        class_map = {}

        for state in self.ag.states:
            class_map[state.id_] = [1]

        with open("graphsage_input/{}-class_map.json".format(self.prefix),
                  "w") as f:
            json.dump(class_map, f, indent=2)

    def create_feats(self):
        """
        Create the file {prefix}-feats.npy which lists the features of the
        nodes.
        """
        feats = np.zeros((self.ag.N, len(self.ag.propositions)))

        for i in range(self.ag.N):
            state = self.ag.states[i]
            for id_proposition in state.ids_propositions:
                feats[i, self.ag.proposition_mapping[id_proposition]] = 1

        np.save("graphsage_input/{}-feats.npy".format(self.prefix), feats)

    def create_walks(self, walk_len=5, n_walks=50):
        """
        Create the file {prefix}-walks.txt which is a list of co-occurences
        between the nodes obtained through random walks.

        :param int walk_len: The length of random walks.
        :param int n_walks: The number of random walks to perform for each
        node.
        """
        # TODO: change to take into account test and val nodes
        pairs = []
        for id_ in range(self.ag.N):
            state = self.ag.states[id_]

            # If the state is isolated, it is impossible to perform random
            # walks
            if not state.in_ and not state.out:
                continue

            # Perform n_walks random walks
            for i in range(n_walks):
                current_state_id = id_
                current_state = state

                # Perform a random walk
                for j in range(walk_len):
                    ids_neighbours = np.concatenate(
                        [list(current_state.in_),
                         list(current_state.out)])
                    next_state_id = int(np.random.choice(ids_neighbours))

                    # Don't save self occurences
                    if current_state_id != id_:
                        pairs.append((id_, current_state_id))

                    current_state_id = next_state_id
                    current_state = self.ag.states[current_state_id]

        with open("graphsage_input/{}-walks.txt".format(self.prefix),
                  "w") as f:
            f.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))


class Graphsage(Embedding):
    """
    Algorithm that creates embeddings of the nodes in a graph by training
    aggregator functions. It was invented by Hamilton et al.

    :param AttackGraph ag: The attack graph.
    :param int dim_embedding: The dimension of the embedding of each node.
    :param str prefix: The prefix of each one of the input files that should be
    created.
    """
    def __init__(self, ag: AttackGraph, dim_embedding: int, prefix: str):
        super().__init__(ag, dim_embedding)

        self.ag = ag
        self.dim_embedding = dim_embedding
        self.prefix = prefix

        self.client = docker.from_env()
        self.container = None
        self.run_container()

    def run(self, size_layer_1=16):
        """
        Run the GraphSAGE algorithm.

        :param int size_layer_1: Size of the hidden layer.
        """
        # Create the input files
        fc = FileCreator(self.ag, self.prefix)
        fc.create_files()

        # Run the container
        self.run_container()

        # Transfer the input files
        self.transfer_files()

        # Run Graphsage
        # The size of the output layer should be equal to half the size of
        # dim_embedding if the aggregator is concatenating
        self.run_graphsage(size_layer_1=size_layer_1,
                           size_layer_2=self.dim_embedding / 2)

        # Create the embeddings from the output files
        self.create_embeddings()

    def run_container(self):
        """
        Get the reference of an existing GraphSAGE container or create a new
        one.
        """
        for container in self.client.containers.list():
            if container.attrs["Config"]["Image"] == "graphsage":
                self.container = container

        if not self.container:
            self.container = self.client.containers.run("graphsage",
                                                        detach=True)

    def transfer_files(self):
        """
        Transfer the input files to the GraphSAGE container.
        The input files must exist.
        """
        Path("temp").mkdir(exist_ok=True)

        file_names = [
            "class_map.json", "feats.npy", "G.json", "id_map.json", "walks.txt"
        ]

        # Create an archive with the input files
        with tarfile.open("temp/{}.tar".format(self.prefix), "w") as f:
            for file_name in file_names:
                f.add("graphsage_input/{}-{}".format(self.prefix, file_name))

        # Run the container
        self.run_container()

        # Transfer the archive to the container
        data = open("temp/{}.tar".format(self.prefix), "rb").read()
        self.container.put_archive("/notebooks", data)

    def run_graphsage(self, size_layer_1=16, size_layer_2=8):
        """
        Run GraphSAGE in the container.

        :param int size_layer_1: The size of the hidden layer.
        :param int size_layer_2: The size of the output layer.
        """
        prefix_path = "./graphsage_input/{}".format(self.prefix)

        # Build the command line
        # Start by choosing the unsupervised model
        command = "python -m graphsage.unsupervised_train "

        # Add the prefix of the input files
        command += "--train_prefix {} ".format(prefix_path)

        # Specify the type of aggregator
        command += "--model graphsage_mean "

        # Add the size of the layers
        command += "--dim_1 {} --dim_2 {} ".format(size_layer_1, size_layer_2)

        # Add various parameters
        command += "--max_total_steps 1000 --validate_iter 10"

        self.container.exec_run(command)

    def create_embeddings(self):
        """
        Create the embeddings from the output files.
        """
        # Create a folder for Graphsage results
        Path("graphsage_output").mkdir(exist_ok=True)

        # Find the path to the result files
        result_folders = self.container.exec_run(
            "ls unsup-graphsage_input -t").output.decode("utf-8")

        # Split according to line breaks and remove the last entry which is
        # always empty
        result_folders = result_folders.split("\n")[:-1]

        # Take the most recent result folder
        folder = result_folders[0]

        # Copy the folder to a tar file
        path = "/notebooks/unsup-graphsage_input/{}".format(folder)
        stream, stats = self.container.get_archive(path)

        tar_file_name = "temp/{}.tar".format(stats["name"])

        with open(tar_file_name, "wb") as f:
            for chunk in stream:
                f.write(chunk)

        # Extract the content of the tar file
        files_to_copy = ["val.txt", "val.npy"]
        for i in range(len(files_to_copy)):
            files_to_copy[i] = "{}/{}".format(folder, files_to_copy[i])

        with tarfile.open(tar_file_name) as f:
            members = []

            # Only keep the basename of the files
            for member in f.getmembers():
                if member.name in files_to_copy:
                    member.path = PurePath(member.path).name
                    members.append(member)

            f.extractall("graphsage_output", members)

        # Extract information from the files
        with open("graphsage_output/val.txt") as f:
            order = [int(i) for i in f.read().split("\n")]

        embeddings = np.load("graphsage_output/val.npy")
        sorted_embeddings = np.zeros_like(embeddings)
        for i in range(len(embeddings)):
            sorted_embeddings[order[i]] = embeddings[i]

        self.embeddings = sorted_embeddings
