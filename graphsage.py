import json
import numpy as np
from pathlib import Path
from mulval import MulvalAttackGraph
from attack_graph import AttackGraph


class FileCreator:
    def __init__(self, ag: AttackGraph, prefix: str):
        self.ag = ag
        self.prefix = prefix

    def create_files(self):
        Path("graphsage_input").mkdir(exist_ok=True)

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
        id_map = {}

        for i in range(self.ag.N):
            state = self.ag.states[i]
            id_map[state.id_] = i

        with open("graphsage_input/{}-id_map.json".format(self.prefix),
                  "w") as f:
            json.dump(id_map, f, indent=2)

    def create_class_map(self):
        class_map = {}

        for state in self.ag.states:
            class_map[state.id_] = [1]

        with open("graphsage_input/{}-class_map.json".format(self.prefix),
                  "w") as f:
            json.dump(class_map, f, indent=2)

    def create_feats(self):
        feats = np.zeros((self.ag.N, len(self.ag.propositions)))

        for i in range(self.ag.N):
            state = self.ag.states[i]
            for id_proposition in state.ids_propositions:
                feats[i, self.ag.proposition_mapping[id_proposition]] = 1

        np.save("graphsage_input/{}-feats.npy".format(self.prefix), feats)

    def create_walks(self, walk_len=5, n_walks=50):
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


mag = MulvalAttackGraph()
mag.parse_from_file("./mulval_data/AttackGraph.xml")
ag = AttackGraph()
ag.import_mulval_attack_graph(mag)

fc = FileCreator(ag, "toy")
fc.create_files()
