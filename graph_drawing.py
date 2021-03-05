from mulval import MulvalAttackGraph, MulvalVertexType
from attack_graph import AttackGraph
import pydot


class GraphDrawer:
    """
    Basic class for drawing graphs.
    """
    @staticmethod
    def save_graph_to_file(graph: pydot.Graph, extension: str):
        """
        Save a pydot graph in a file.

        :param pydot.Graph graph: The graph to save.
        :param str extension: The extension of the file.
        """
        file_name = "./output/" + str(graph.get_name()) + "." + extension
        if extension == "dot":
            graph.write_raw(file_name)
        elif extension == "pdf":
            graph.write_pdf(file_name)
        elif extension == "png":
            graph.write_png(file_name)
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
        :return pydot.Graph graph: The built graph.
        """
        if not self.mag.vertices:
            return

        graph = pydot.Dot(graph_name=graph_name, graph_type="digraph")

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

            graph.add_node(node)

        # Add the edges to the graph
        for id_i in self.mag.ids:
            vertex_i = self.mag.vertices[id_i]
            for id_j in vertex_i.out:
                edge = pydot.Edge(id_i, id_j)
                graph.add_edge(edge)

        return graph


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
        :return pydot.Graph graph: The built graph.
        """
        if not self.ag.states:
            return

        graph = pydot.Dot(graph_name=graph_name, graph_type="digraph")

        # Find whether or not a ranking has been applied
        ranking = hasattr(self.ag.states[0], "ranking_score")

        # If a ranking has been applied, find the extrema of the values
        if ranking:
            min_, max_ = self.find_extrema_ranking_values()

        # Add the states to the graph
        for state in self.ag.states:
            # Create the node
            node = pydot.Node(name=state.id_,
                              label=str(state.id_),
                              shape="circle")

            # If a ranking has been applied, change the color of the node
            # accordingly
            if ranking:
                saturation = (state.ranking_score - min_) / (max_ - min_)
                color = "0 " + str(round(saturation, 3)) + " 1"
                node.set_style("filled")
                node.set_fillcolor(color)

            graph.add_node(node)

        # Add the transitions to the graph
        for transition in self.ag.transitions:
            edge = pydot.Edge(transition.state_id_from, transition.state_id_to)
            if labels:
                edge.set_label(transition.id_)
            graph.add_edge(edge)

        return graph

    def find_extrema_ranking_values(self):
        """
        Find the minimum and the maximum of the ranking values among the
        states.
        The function should be called only if a ranking method has been
        applied.

        :return tuple (min, max): The minimum and the maximum of the ranking
        values.
        """
        min_ = self.ag.states[0].ranking_score
        max_ = self.ag.states[0].ranking_score
        for state in self.ag.states:
            ranking_score = state.ranking_score
            if ranking_score < min_:
                min_ = ranking_score
            elif ranking_score > max_:
                max_ = ranking_score
        return min_, max_
