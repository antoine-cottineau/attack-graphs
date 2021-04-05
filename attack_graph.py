import bisect
import networkx as nx
import pathlib
import pydot
import xml.etree.ElementTree as ET


class BaseGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()

    def draw(self, path: str):
        # Create the parent directory(ies) of the file
        file_ = pathlib.Path(path)
        parent_directory = file_.parent
        parent_directory.mkdir(exist_ok=True, parents=True)

        # Get the extension of the file
        extension = file_.suffix

        # Draw the graph
        pydot_graph = self._convert_to_pydot()
        if extension == ".dot":
            pydot_graph.write_raw(file_)
        elif extension == ".pdf":
            pydot_graph.write_pdf(file_)
        elif extension == ".png":
            pydot_graph.write_png(file_)
        else:
            raise Exception("File type {} not supported.".format(extension))

    def import_from_mulval_xml_file(self, path: str):
        pass

    def _convert_to_pydot(self):
        return nx.nx_pydot.to_pydot(self)


class MulvalAttackGraph(BaseGraph):
    def __init__(self):
        super().__init__()

    def import_from_mulval_xml_file(self, path: str):
        # Parse the file
        tree = ET.parse(path)
        root = tree.getroot()
        nodes = root.findall(path="vertices/vertex")
        edges = root.findall(path="arcs/arc")

        # Add nodes to the instance
        for i, node in enumerate(nodes):
            fact = node.find("fact").text
            metric = int(node.find("metric").text)
            type_ = node.find("type").text

            shape = "box"
            if type_ == "AND":
                shape = "oval"
            elif type_ == "OR":
                shape = "diamond"

            self.add_node(i,
                          fact=fact,
                          metric=metric,
                          type=type_,
                          label="{}: {}".format(i, fact),
                          shape=shape)

        # Add edges to the instance
        for edge in edges:
            # MulVAL starts the indexing of the nodes at 1 but we start at 0
            src = int(edge.find("src").text) - 1
            dst = int(edge.find("dst").text) - 1

            # Source and destination nodes seem inverted in the attack graph
            # files generated by MulVAL
            self.add_edge(dst, src)


class AttackGraph(BaseGraph):
    def __init__(self):
        super().__init__()

        self.propositions = {}

    def import_from_mulval_xml_file(self, path: str):
        mag = MulvalAttackGraph()
        mag.import_from_mulval_xml_file(path)
        self.import_from_mulval_attack_graph(mag)

    def import_from_mulval_attack_graph(self, mag: MulvalAttackGraph):
        ids_initial_propositions = []
        ids_edges = []

        for i, node in mag.nodes(data=True):
            if node["type"] == "LEAF":
                # The vertex is a LEAF which corresponds to a proposition that
                # is true at the beginning
                ids_initial_propositions.append(i)
                self.propositions[i] = (i, node["fact"])
            elif node["type"] == "OR":
                # The vertex is an OR which corresponds to a proposition that
                # is false at the beginning
                self.propositions[i] = (i, node["fact"])
            else:
                # The vertex is an AND which corresponds to an edge
                ids_edges.append(i)

        # Create the initial node
        initial_node = (0, {"ids_propositions": ids_initial_propositions})
        self.add_nodes_from([initial_node])

        # Fill the graph
        self._fill_graph_recursively(mag, initial_node, ids_edges)

        # Create a mapping between the id of each proposition and an integer
        # between 0 and len(self.propositions) - 1
        self._create_proposition_mapping()

    def update_colors_based_on_ranking(self):
        for _, node in self.nodes(data=True):
            # The color saturation of a node is calculated thanks to a linear
            # interpolation between the two extrema of the ranking
            saturation = (node["ranking_score"] - self.ranking_min) / (
                self.ranking_max - self.ranking_min)
            color = "0 " + str(round(saturation, 3)) + " 1"
            node["style"] = "filled"
            node["fillcolor"] = color

    def _create_proposition_mapping(self):
        proposition_mapping = {}
        ids_propositions = [*self.propositions]
        for i in range(len(self.propositions)):
            proposition = self.propositions[ids_propositions[i]]
            proposition_mapping[proposition[0]] = i
        self.proposition_mapping = proposition_mapping

    def _convert_to_pydot(self):
        pydot_graph = nx.nx_pydot.to_pydot(self)

        if "id_cluster" not in self.nodes(data=True)[0]:
            return pydot_graph

        # Create clusters and add the existing nodes to them
        clusters = {}

        for i, node in self.nodes(data=True):
            id_cluster = node["id_cluster"]

            # If the cluster does not exist, create a new one
            if id_cluster not in clusters:
                clusters[id_cluster] = pydot.Subgraph(
                    "cluster_{}".format(id_cluster))

            # Add the pydot node to the cluster
            clusters[id_cluster].add_node(pydot_graph.get_node(str(i))[0])

        # Create a new pydot graph
        new_pydot_graph = pydot.Dot()

        # Add the clusters to the new graph
        for id_cluster in clusters:
            new_pydot_graph.add_subgraph(clusters[id_cluster])

        # Add the edges to the new graph
        for edge in pydot_graph.get_edge_list():
            new_pydot_graph.add_edge(edge)

        return new_pydot_graph

    def _fill_graph_recursively(self, mag: MulvalAttackGraph, node: tuple,
                                ids_edges: list):
        current_ids_propositions = node[1]["ids_propositions"]
        # Loop through all the edges
        for id_edge in ids_edges:
            # Check if this edge can be used depending on the current true
            # propositions
            is_possible = True
            for id_required in mag.predecessors(id_edge):
                is_possible &= id_required in current_ids_propositions

            if not is_possible:
                continue

            # Get the proposition that following this edge grants
            id_granted_proposition = list(mag.successors(id_edge))[0]

            # Add the proposition if it isn't already granted
            new_ids_propositions = current_ids_propositions.copy()
            if id_granted_proposition not in current_ids_propositions:
                bisect.insort(new_ids_propositions, id_granted_proposition)

            # Search if a node with such true propositions already exists
            similar_nodes = [
                i
                for i, ids_propositions in self.nodes(data="ids_propositions")
                if ids_propositions == new_ids_propositions
            ]
            if similar_nodes:
                # Search if there is already an edge between the source node
                # and the reached one
                similar_edges = [(src, dst) for src, dst in self.edges()
                                 if src == node[0] and dst == id_edge]
                if not similar_edges and node[0] != similar_nodes[0]:
                    # Just add the edge
                    self.add_edge(node[0], similar_nodes[0])
            else:
                # Create a brand new node
                new_node = (self.number_of_nodes(), {
                    "ids_propositions": new_ids_propositions
                })
                self.add_nodes_from([new_node])

                # Add an edge
                self.add_edge(node[0], new_node[0])

                # Call recursively this function with the new node and and with
                # the used edge removed
                new_ids_edges = ids_edges.copy()
                new_ids_edges.remove(id_edge)
                self._fill_graph_recursively(mag, new_node, new_ids_edges)
