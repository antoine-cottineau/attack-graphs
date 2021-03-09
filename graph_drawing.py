from mulval import MulvalAttackGraph, MulvalVertexType
from attack_graph import AttackGraph
import pydot


class GraphDrawer:
    """
    Base class for drawing graphs.
    """
    def save_graph_to_file(self, extension: str):
        """
        Save a graph in a file.

        :param str extension: The extension of the file.
        """
        file_name = "./output/" + str(self.graph.get_name()) + "." + extension
        if extension == "dot":
            self.graph.write_raw(file_name)
        elif extension == "pdf":
            self.graph.write_pdf(file_name)
        elif extension == "png":
            self.graph.write_png(file_name)
        else:
            print("File type not supported.")


class MulvalGraphDrawer(GraphDrawer):
    """
    Tool to draw a MulVAL graph thanks to pydot.

    :param MulvalAttackGraph mag: The graph to draw.
    """
    def __init__(self, mag: MulvalAttackGraph):
        self.mag = mag

    def create_pydot_graph(self, graph_name: str, simplified=False):
        """
        Create a pydot graph from the provided MulVAL attack graph.

        :param str graph_name: The name of the graph.
        :param str simplified: Whether or not the labels should be simplified.
        """
        if not self.mag.vertices:
            return

        self.graph = pydot.Dot(graph_name=graph_name, graph_type="digraph")

        # Add the vertices to the graph
        for id_ in self.mag.ids:
            vertex = self.mag.vertices[id_]

            # Build the label of the node
            label = str(vertex.id_) + ":" + vertex.fact + ":" + str(
                vertex.metric)
            if simplified:
                label = str(vertex.id_)

            # Change the shape of the node
            shape = "box"
            if vertex.type_ == MulvalVertexType.AND:
                shape = "ellipse"
            elif vertex.type_ == MulvalVertexType.OR:
                shape = "diamond"

            # Create the node with all the above parameters
            node = pydot.Node(name=vertex.id_, label=label, shape=shape)

            self.graph.add_node(node)

        # Add the edges to the graph
        for id_i in self.mag.ids:
            vertex_i = self.mag.vertices[id_i]
            for id_j in vertex_i.out:
                edge = pydot.Edge(id_i, id_j)
                self.graph.add_edge(edge)


class AttackGraphDrawer(GraphDrawer):
    """
    Tool to draw an attack graph thanks to pydot.

    :param AttackGraph ag: The graph to draw.
    """
    def __init__(self, ag: AttackGraph):
        self.ag = ag

    def create_pydot_graph(self, graph_name: str, labels=False):
        """
        Create a pydot graph from the provided attack graph.

        :param str graph_name: The name of the graph.
        :param bool labels: Whether or not the edge labels should be shown.
        """
        if not self.ag.states:
            return

        self.graph = pydot.Dot(graph_name=graph_name, graph_type="digraph")

        # Find whether or not ranking has been applied
        self.ranking = hasattr(self.ag.states[0], "ranking_score")

        # Find whether or not clustering has been applied
        self.clustering = hasattr(self.ag.states[0], "id_cluster")

        # Create a dictionnary matching the cluster ids and the lists of states
        # in the clusters.
        self.clusters = {}

        # Add the states to the graph and complete the clusters
        for state in self.ag.states:
            self.create_pydot_node(state)

        # If clustering has been applied, add the clusters to the graph
        if self.clustering:
            for id_cluster in self.clusters:
                self.graph.add_subgraph(self.clusters[id_cluster])

        # Add the transitions to the graph
        for transition in self.ag.transitions:
            edge = pydot.Edge(transition.state_id_from, transition.state_id_to)
            if labels:
                edge.set_label(transition.id_)
            self.graph.add_edge(edge)

    def create_pydot_node(self, state):
        """
        Create a pydot node from the provided state. Also add the node to the
        corresponding cluster if needed.

        :param State state: The state to convert.
        """
        node = pydot.Node(name=state.id_, label=str(state.id_), shape="circle")

        # If ranking has been applied, change the color of the node accordingly
        if self.ranking:
            saturation = (state.ranking_score - self.ag.ranking_min) / (
                self.ag.ranking_max - self.ag.ranking_min)
            color = "0 " + str(round(saturation, 3)) + " 1"
            node.set_style("filled")
            node.set_fillcolor(color)

        # If clustering has been applied, add the node to the corresponding
        # cluster
        if self.clustering:
            id_cluster = state.id_cluster

            # If the cluster does not exist, create a new one
            if id_cluster not in self.clusters:
                self.clusters[id_cluster] = pydot.Subgraph("cluster_" +
                                                           str(id_cluster))

            # Add the new node to the cluster
            self.clusters[id_cluster].add_node(node)
        else:
            self.graph.add_node(node)
