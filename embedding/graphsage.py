import json
import networkx as nx
import numpy as np

from attack_graph import AttackGraph
from docker_handler import DockerHandler
from embedding.embedding import Embedding
from pathlib import Path


class FileCreator:
    def __init__(self, ag: AttackGraph, prefix: str):
        self.ag = ag
        self.prefix = prefix

        self.base_folder = "methods_input/graphsage"

    def create_files(self):
        Path(self.base_folder).mkdir(exist_ok=True, parents=True)

        self.create_G()
        self.create_id_map()
        self.create_class_map()
        self.create_feats()
        self.create_walks()

    def create_G(self):
        G = {
            "directed": False,
            "graph": {
                "name": "graph"
            },
            "nodes": [],
            "links": []
        }

        for i in self.ag.nodes():
            node = {
                "feature": [],
                "id": i,
                "label": [1],
                "test": False,
                "val": False
            }
            G["nodes"].append(node)

        for (src, dst) in self.ag.edges():
            link = {
                "source": src,
                "target": dst,
                "test_removed": False,
                "train_removed": False
            }
            G["links"].append(link)

        with open("{}/{}-G.json".format(self.base_folder, self.prefix),
                  "w") as f:
            json.dump(G, f, indent=2)

    def create_id_map(self):
        id_map = {}

        for i in self.ag.nodes():
            id_map[i] = i

        with open("{}/{}-id_map.json".format(self.base_folder, self.prefix),
                  "w") as f:
            json.dump(id_map, f, indent=2)

    def create_class_map(self):
        class_map = {}

        for i in self.ag.nodes():
            class_map[i] = [1]

        with open("{}/{}-class_map.json".format(self.base_folder, self.prefix),
                  "w") as f:
            json.dump(class_map, f, indent=2)

    def create_feats(self):
        feats = np.zeros(
            (self.ag.number_of_nodes(), len(self.ag.propositions)))

        for i, node in self.ag.nodes(data=True):
            for id_proposition in node["ids_propositions"]:
                feats[i, self.ag.get_node_mapping()[id_proposition]] = 1

        np.save("{}/{}-feats.npy".format(self.base_folder, self.prefix), feats)

    def create_walks(self, walk_len=5, n_walks=50):
        # TODO: change to take into account test and val nodes
        pairs = []
        for id_ in self.ag.nodes():

            # If the node is isolated, it is impossible to perform random
            # walks
            if not nx.all_neighbors(self.ag, id_):
                continue

            # Perform n_walks random walks
            for i in range(n_walks):
                current_state_id = id_

                # Perform a random walk
                for j in range(walk_len):
                    ids_neighbours = list(
                        nx.all_neighbors(self.ag, current_state_id))
                    next_state_id = int(np.random.choice(ids_neighbours))

                    # Don't save self occurences
                    if current_state_id != id_:
                        pairs.append((id_, current_state_id))

                    current_state_id = next_state_id

        with open("{}/{}-walks.txt".format(self.base_folder, self.prefix),
                  "w") as f:
            f.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))


class Graphsage(Embedding):
    def __init__(self, ag: AttackGraph, dim_embedding: int, prefix: str):
        super().__init__(ag, dim_embedding)

        self.prefix = prefix

        self.dh = DockerHandler("graphsage")
        self.input_folder = "methods_input/graphsage"
        self.output_folder = "methods_output/graphsage"

    def run(self, size_layer_1=16):
        # Create the input files
        fc = FileCreator(self.ag, self.prefix)
        fc.create_files()

        # Run the container
        self.dh.run_container()

        # Transfer the input files
        self.dh.transfer_folder(self.input_folder, "/notebooks", self.prefix)

        # Run Graphsage
        # The size of the output layer should be equal to half the size of
        # dim_embedding if the aggregator is concatenating
        self.run_graphsage(size_layer_1=size_layer_1,
                           size_layer_2=self.dim_embedding // 2)

        # Create the embedding from the output files
        self.create_embedding()

        # Save the embedding
        self.save_embedding_in_file("{}/embedding.npy".format(
            self.output_folder))

    def run_graphsage(self, size_layer_1=16, size_layer_2=8):
        prefix_path = "./{}/{}".format(self.input_folder, self.prefix)

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

        self.dh.run_command(command)

    def create_embedding(self):
        # Find the path to the result files and extract the most recent one
        folder = self.dh.list_elements_in_container("unsup-graphsage")[0]

        # Copy the folder to a tar file
        container_path = "/notebooks/unsup-graphsage/{}".format(folder)
        file_filter = ["val.txt", "val.npy"]

        self.dh.copy_folder_from_container(container_path, self.output_folder,
                                           file_filter)

        # Extract information from the files
        with open("{}/val.txt".format(self.output_folder)) as f:
            order = [int(i) for i in f.read().split("\n")]

        embedding = np.load("{}/val.npy".format(self.output_folder))
        sorted_embedding = np.zeros_like(embedding)
        for i in range(len(embedding)):
            sorted_embedding[order[i]] = embedding[i]

        self.embedding = sorted_embedding
