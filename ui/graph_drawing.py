from metrics.homer import RiskQuantifier
from metrics.sheyner import ValueIteration
import dash_core_components as dcc
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import ui.constants
from attack_graph import DependencyAttackGraph, StateAttackGraph
from ranking.mehta import PageRankMethod, KuehlmannMethod
from typing import Dict, List, Tuple


class BaseGraphDrawer:
    def __init__(self, parameters: dict):
        self.parameters = parameters

        self.scatter_objects: List[go.Scatter] = []

    def apply(self) -> dcc.Graph:
        self.parse_parameters()
        self.prune_graph()
        self.node_positions = self.compute_node_positions()
        self.ranking_values = self.apply_ranking()
        self.ranking_order = self.compute_ranking_order()
        self.add_node_scatter_objects()
        self.add_edge_scatter_objects()
        graph = self.build_graph()
        return graph

    def parse_parameters(self):
        # Set default values
        self.selected_exploits_ids: List[int] = None
        self.ranking_method: str = None
        self.clustering_method: str = None

        if not self.parameters:
            return

        # Parse the exploits to keep during pruning
        if "selected_exploits" in self.parameters:
            self.selected_exploits_ids = [
                int(i) for i in self.parameters["selected_exploits"]
            ]

        # Parse the ranking method
        if "ranking_method" in self.parameters and self.parameters[
                "ranking_method"] != "none":
            self.ranking_method = self.parameters["ranking_method"]

        # Parse the clustering method
        if "clustering_method" in self.parameters and self.parameters[
                "clustering_method"] != "none":
            self.clustering_method = self.parameters["clustering_method"]

    def prune_graph(self):
        pass

    def apply_ranking(self) -> Dict[int, float]:
        return None

    def compute_ranking_order(self) -> Dict[int, int]:
        if self.ranking_method is None:
            return None

        ranking_order = {}
        argsort = np.argsort(list(self.ranking_values.values()))
        n = len(self.ranking_values)
        for i in range(n):
            node = list(self.ranking_values)[argsort[i]]
            ranking_order[node] = n - i

        return ranking_order

    def build_graph(self) -> dcc.Graph:
        axis_parameters = dict(showgrid=False,
                               zeroline=False,
                               showticklabels=False)

        figure = go.Figure(data=self.scatter_objects,
                           layout=go.Layout(
                               margin=dict(b=8, l=8, r=8, t=8),
                               showlegend=False,
                               paper_bgcolor=ui.constants.color_dark,
                               plot_bgcolor=ui.constants.color_dark,
                               xaxis=axis_parameters,
                               yaxis=axis_parameters))

        return dcc.Graph(id="graph",
                         figure=figure,
                         config=dict(displaylogo=False))

    def add_node_scatter_object(self,
                                dict_positions: Dict[int, Tuple[float, float]],
                                dict_hovertexts: Dict[int, str],
                                symbol: str = "circle",
                                ranking_legend_text: str = "Ranking score"):
        x = []
        y = []
        hovertext = []
        ranking_colors = []
        for node in dict_positions:
            position = dict_positions[node]
            x.append(position[0])
            y.append(position[1])
            hovertext.append(dict_hovertexts[node])
            if self.ranking_method is not None:
                ranking_colors.append(self.ranking_values[node])

        scatter_object = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            hoverinfo="text",
            hovertext=hovertext,
            hoverlabel=dict(font=dict(family="Montserrat")),
            marker=dict(size=12,
                        color=ui.constants.color_accent,
                        symbol=symbol))

        # Update the color of the nodes if ranking has been applied
        if self.ranking_method is not None:
            font = dict(family="Montserrat", color=ui.constants.color_light)
            scatter_object.marker = dict(
                size=12,
                showscale=True,
                colorscale="Reds",
                color=ranking_colors,
                colorbar=dict(tickfont=font,
                              title=dict(text=ranking_legend_text, font=font)))

        self.scatter_objects.append(scatter_object)

    def add_edge_scatter_object(self, edges: List[Tuple[Tuple[float, float],
                                                        Tuple[float, float]]]):
        x = []
        y = []
        for edge in edges:
            src = edge[0]
            dst = edge[1]
            x += [src[0], dst[0], None]
            y += [src[1], dst[1], None]

        self.scatter_objects.append(
            go.Scatter(x=x,
                       y=y,
                       mode="lines",
                       line=dict(width=1.5, color=ui.constants.color_light)))

    def compute_node_positions(self) -> Dict[int, Tuple[float, float]]:
        return None

    def add_node_scatter_objects(self):
        pass

    def add_edge_scatter_objects(self):
        pass


class StateAttackGraphDrawer(BaseGraphDrawer):
    def __init__(self, graph: StateAttackGraph, parameters: dict):
        super().__init__(parameters)

        self.graph = graph

    def prune_graph(self):
        if self.selected_exploits_ids is not None:
            self.graph = self.graph.get_pruned_graph(
                self.selected_exploits_ids)

    def compute_node_positions(self) -> Dict[int, Tuple[float, float]]:
        n_initial_propositions = len(self.graph.nodes[0]["ids_propositions"])

        for node, ids_propositions in self.graph.nodes(
                data="ids_propositions"):
            self.graph.nodes[node]["subset"] = len(
                ids_propositions) - n_initial_propositions

        node_positions = nx.drawing.layout.multipartite_layout(self.graph)
        node_positions = dict([(node, (float(position[0]), float(position[1])))
                               for node, position in node_positions.items()])

        return node_positions

    def apply_ranking(self) -> Dict[int, float]:
        if self.ranking_method is None:
            return None

        if self.ranking_method == "pagerank":
            return PageRankMethod(self.graph).apply()
        elif self.ranking_method == "kuehlmann":
            return KuehlmannMethod(self.graph).apply()
        elif self.ranking_method == "vi":
            return ValueIteration(self.graph).apply()[0]
        else:
            return None

    def add_node_scatter_objects(self):
        hovertexts: Dict[int, str] = {}

        for node in self.graph.nodes:
            hovertext = "id: {}".format(node)

            if self.ranking_method is not None:
                hovertext += "<br>ranking position: {}/{}".format(
                    self.ranking_order[node], self.graph.number_of_nodes())
                hovertext += "<br>ranking value: {:0.2E}".format(
                    self.ranking_values[node])

            hovertexts[node] = hovertext

        self.add_node_scatter_object(self.node_positions, hovertexts)

    def add_edge_scatter_objects(self):
        edges = []

        for edge in self.graph.edges:
            src = self.node_positions[edge[0]]
            dst = self.node_positions[edge[1]]
            edges.append((src, dst))

        self.add_edge_scatter_object(edges)


class DependencyAttackGraphDrawer(BaseGraphDrawer):
    def __init__(self, graph: DependencyAttackGraph, parameters: dict) -> None:
        super().__init__(parameters)

        self.graph = graph

    def prune_graph(self):
        if self.selected_exploits_ids is not None:
            self.graph = self.graph.get_pruned_graph(
                self.selected_exploits_ids)

    def compute_node_positions(self) -> Dict[int, Tuple[float, float]]:
        node_layers: Dict[int, int] = {}
        nodes_to_evaluate: List[int] = []

        # Find the final node
        final_nodes = [
            node for node in self.graph.nodes
            if len(list(self.graph.successors(node))) == 0
        ]
        nodes_to_evaluate += final_nodes

        while len(node_layers) < self.graph.number_of_nodes():
            node = nodes_to_evaluate[0]
            data = self.graph.nodes[node]
            successors = set(self.graph.successors(node))

            layer = None
            if node in final_nodes:
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
                for predecessor in self.graph.predecessors(node):
                    if predecessor not in nodes_to_evaluate:
                        nodes_to_evaluate.append(predecessor)

        for node in self.graph.nodes:
            self.graph.nodes[node]["subset"] = max(
                node_layers.values()) - node_layers[node]

        node_positions = nx.drawing.layout.multipartite_layout(self.graph)
        node_positions = dict([(node, (float(position[0]), float(position[1])))
                               for node, position in node_positions.items()])

        return node_positions

    def apply_ranking(self) -> Dict[int, float]:
        if self.ranking_method is None:
            return None

        if self.ranking_method == "homer":
            return RiskQuantifier(self.graph).apply()
        else:
            return None

    def add_node_scatter_objects(self):
        propositions_positions: Dict[int, Tuple[float, float]] = {}
        propositions_hovertext: Dict[int, str] = {}

        exploits_positions: Dict[int, Tuple[float, float]] = {}
        exploits_hovertext: Dict[int, str] = {}

        for node, data in self.graph.nodes(data=True):
            position = self.node_positions[node]
            hovertext = "id: {}".format(node)
            if "id_proposition" in data:
                propositions_positions[node] = position
                hovertext += "<br>proposition id: {}".format(
                    data["id_proposition"])
                propositions_hovertext[node] = hovertext
            else:
                exploits_positions[node] = position
                hovertext += "<br>exploit id: {}".format(data["id_exploit"])
                exploits_hovertext[node] = hovertext

        self.add_node_scatter_object(propositions_positions,
                                     propositions_hovertext,
                                     symbol="diamond",
                                     ranking_legend_text="Probability")

        self.add_node_scatter_object(exploits_positions,
                                     exploits_hovertext,
                                     ranking_legend_text="Probability")

    def add_edge_scatter_objects(self):
        edges = []

        for edge in self.graph.edges:
            src = self.node_positions[edge[0]]
            dst = self.node_positions[edge[1]]
            edges.append((src, dst))

        self.add_edge_scatter_object(edges)
