import dash_core_components as dcc
import networkx as nx
import plotly.graph_objects as go
import ui.constants
from attack_graph import DependencyAttackGraph, StateAttackGraph
from typing import Dict, List, Tuple
from ui.cluster_drawing import ClusterDrawer


class BaseGraphDrawer:
    def __init__(self, parameters: dict):
        self.parameters = parameters

        self.objects: List[go.Scatter] = []
        self.selected_exploits: List[int] = None
        self.clusters: Dict[str, dict] = None
        self.positions: Dict[int, Tuple[float, float]] = None
        self.zones: List[dict] = None

    def apply(self) -> dcc.Graph:
        self.parse_parameters()
        self.prune_graph()
        self.compute_positions()
        self.compute_cluster_drawing()
        self.add_all_objects()
        return self.build_graph()

    def parse_parameters(self):
        if not self.parameters:
            return

        self.selected_exploits = self.parameters["selected_exploits"]
        self.clusters = self.parameters["clusters"]

    def prune_graph(self):
        pass

    def compute_positions(self):
        return None

    def compute_cluster_drawing(self):
        if self.clusters is None:
            return

        cluster_drawer = ClusterDrawer(self.positions, self.clusters)
        cluster_drawer.apply()
        self.positions = cluster_drawer.positions
        self.zones = cluster_drawer.zones

    def add_all_objects(self):
        self.add_zone_objects()

    def add_node_objects(self, dict_positions: Dict[int, Tuple[float, float]],
                         dict_hovertexts: Dict[int, str], colors: List[str]):
        x = []
        y = []
        hovertext = []
        for node in dict_positions:
            position = dict_positions[node]
            x.append(position[0])
            y.append(position[1])
            hovertext.append(dict_hovertexts[node])

        node_objects = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            hoverinfo="text",
            hovertext=hovertext,
            hoverlabel=dict(font=dict(family="Montserrat")),
            marker=dict(size=12, color=colors))

        self.objects.append(node_objects)

    def add_edge_objects(self, edges: List[Tuple[Tuple[float, float],
                                                 Tuple[float, float]]]):
        x = []
        y = []
        for edge in edges:
            src = edge[0]
            dst = edge[1]
            x += [src[0], dst[0], None]
            y += [src[1], dst[1], None]

        self.objects.append(
            go.Scatter(x=x,
                       y=y,
                       mode="lines",
                       line=dict(width=1, color="white")))

    def add_zone_objects(self):
        if self.clusters is None:
            return

        for zone in self.zones:
            points = zone["points"]
            i_cluster = zone["cluster"]
            color = self.clusters[i_cluster]["color"]

            x = []
            y = []
            for point in points:
                x.append(point[0])
                y.append(point[1])
            x.append(points[0][0])
            y.append(points[0][1])

            self.objects.append(
                go.Scatter(x=x,
                           y=y,
                           mode="lines",
                           fill="toself",
                           fillcolor=color,
                           line=dict(width=0),
                           hoverinfo="text",
                           text="cluster {}".format(i_cluster)))

    def build_graph(self) -> dcc.Graph:
        axis_parameters = dict(showgrid=False,
                               zeroline=False,
                               showticklabels=False)

        figure = go.Figure(data=self.objects,
                           layout=go.Layout(
                               margin=dict(b=0, l=0, r=0, t=0),
                               paper_bgcolor=ui.constants.color_dark,
                               plot_bgcolor=ui.constants.color_dark,
                               showlegend=False,
                               xaxis=axis_parameters,
                               yaxis=axis_parameters))

        return dcc.Graph(id="object-attack-graph",
                         figure=figure,
                         config=dict(displaylogo=False))


class StateAttackGraphDrawer(BaseGraphDrawer):
    def __init__(self, attack_graph: StateAttackGraph, parameters: dict):
        super().__init__(parameters)

        self.attack_graph = attack_graph

    def prune_graph(self):
        if self.selected_exploits is not None:
            self.attack_graph = self.attack_graph.get_pruned_graph(
                self.selected_exploits)

    def compute_positions(self) -> Dict[int, Tuple[float, float]]:
        if self.attack_graph.number_of_nodes() == 0:
            return {}

        n_initial_propositions = len(
            self.attack_graph.nodes[0]["ids_propositions"])

        for node, ids_propositions in self.attack_graph.nodes(
                data="ids_propositions"):
            self.attack_graph.nodes[node]["subset"] = len(
                ids_propositions) - n_initial_propositions

        self.positions = nx.drawing.layout.multipartite_layout(
            self.attack_graph)
        self.positions = dict([(node, (float(position[0]), float(position[1])))
                               for node, position in self.positions.items()])

    def add_all_objects(self):
        super().add_all_objects()

        # Add the node objects
        hovertexts: Dict[int, str] = {}
        for node in self.attack_graph.nodes:
            hovertext = "id: {}".format(node)
            hovertexts[node] = hovertext

        colors = [ui.constants.color_accent
                  ] * self.attack_graph.number_of_nodes()

        self.add_node_objects(self.positions, hovertexts, colors)

        # Add the edge objects
        edges = []

        for edge in self.attack_graph.edges:
            src = self.positions[edge[0]]
            dst = self.positions[edge[1]]
            edges.append((src, dst))

        self.add_edge_objects(edges)


class DependencyAttackGraphDrawer(BaseGraphDrawer):
    def __init__(self, attack_graph: DependencyAttackGraph,
                 parameters: dict) -> None:
        super().__init__(parameters)

        self.attack_graph = attack_graph

    def prune_graph(self):
        if self.selected_exploits is not None:
            self.attack_graph = self.attack_graph.get_pruned_graph(
                self.selected_exploits)

    def compute_positions(self) -> Dict[int, Tuple[float, float]]:
        node_layers: Dict[int, int] = {}
        nodes_to_evaluate: List[int] = []

        # Find the goal node
        nodes_to_evaluate += self.attack_graph.goal_nodes

        while len(node_layers) < self.attack_graph.number_of_nodes():
            node = nodes_to_evaluate[0]
            data = self.attack_graph.nodes[node]
            successors = set(self.attack_graph.successors(node))

            layer = None
            if node in self.attack_graph.goal_nodes:
                layer = 0
            elif set(node_layers) >= successors:
                # All the successors have been evaluated
                if "id_proposition" in data:
                    layer = max([node_layers[s] for s in successors]) + 1
                else:
                    layer = node_layers[list(successors)[0]] + 1

            # Remove the node from the list of nodes to evaluate
            nodes_to_evaluate = nodes_to_evaluate[1:]

            if layer is None:
                # The node cannot be assigned to a layer yet
                # Add it at the end of the list
                nodes_to_evaluate.append(node)
            else:
                node_layers[node] = layer

                # Add the node's predecessors to the list of nodes to
                # evaluate
                for predecessor in self.attack_graph.predecessors(node):
                    if predecessor not in nodes_to_evaluate:
                        nodes_to_evaluate.append(predecessor)

        for node in self.attack_graph.nodes:
            self.attack_graph.nodes[node]["subset"] = max(
                node_layers.values()) - node_layers[node]

        self.positions = nx.drawing.layout.multipartite_layout(
            self.attack_graph)
        self.positions = dict([(node, (float(position[0]), float(position[1])))
                               for node, position in self.positions.items()])

    def add_all_objects(self):
        super().add_all_objects()

        # Add the node objects
        node_positions: Dict[int, Tuple[float, float]] = {}
        hovertexts: Dict[int, str] = {}
        colors = []

        for node, data in self.attack_graph.nodes(data=True):
            position = self.positions[node]
            hovertext = "id: {}".format(node)
            if "id_proposition" in data:
                node_positions[node] = position
                hovertext += "<br>proposition id: {}".format(
                    data["id_proposition"])
                hovertexts[node] = hovertext
                colors.append(ui.constants.color_accent)
            else:
                node_positions[node] = position
                hovertext += "<br>exploit id: {}".format(data["id_exploit"])
                hovertexts[node] = hovertext
                colors.append(ui.constants.color_accent_secondary)

        self.add_node_objects(node_positions, hovertexts, colors)

        # Add the edge objects
        edges = []

        for edge in self.attack_graph.edges:
            src = self.positions[edge[0]]
            dst = self.positions[edge[1]]
            edges.append((src, dst))

        self.add_edge_objects(edges)
