import dash_core_components as dcc
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import ranking.mehta as ranking
import ui.constants

from attack_graph import AttackGraph
from clustering.drawing import ClusterDrawer
from clustering.white_smyth import Spectral1
from utils import create_random_color


class GraphDrawer:
    def __init__(self, ag: AttackGraph, parameters: dict):
        self.ag = ag
        self.parameters = parameters

    def draw(self) -> dcc.Graph:
        self.prune_graph()
        self.apply_ranking()
        self.compute_node_positions()
        self.create_node_trace()
        self.create_edge_trace()
        self.create_cluster_trace()
        return self.create_graph()

    def prune_graph(self):
        if not self.parameters or "selected_exploits" not in self.parameters:
            return

        ids_exploits = [int(i) for i in self.parameters["selected_exploits"]]
        self.ag = self.ag.get_pruned_graph(ids_exploits)

    def apply_ranking(self):
        if not self.parameters or "ranking_method" not in self.parameters:
            return

        if self.parameters["ranking_method"] == "none":
            return

        # Apply the appropriate ranking method
        if self.parameters["ranking_method"] == "pagerank":
            ranking_values = ranking.PageRankMethod(self.ag).apply()
        else:
            ranking_values = ranking.KuehlmannMethod(self.ag).apply()

        # Compute the ranking order i.e. the position of each node in the
        # ranking
        ranking_order = np.zeros(self.ag.number_of_nodes(), dtype=int)
        indices_sorting = np.flip(np.argsort(list(ranking_values.values())))
        ranking_order[indices_sorting] = np.arange(
            self.ag.number_of_nodes()) + 1
        ranking_order = dict([(id, ranking_order[i])
                              for i, id in enumerate(self.ag.nodes)])

        self.ranking_values = ranking_values
        self.ranking_order = ranking_order

    def compute_node_positions(self):
        n_initial_propositions = len(self.ag.nodes[0]["ids_propositions"])
        for id_node, ids_propositions in self.ag.nodes(
                data="ids_propositions"):
            self.ag.nodes[id_node]["subset"] = len(
                ids_propositions) - n_initial_propositions

        self.positions = nx.drawing.layout.multipartite_layout(self.ag)

    def create_node_trace(self):
        x = []
        y = []
        hovertext = []

        # Create the nodes
        for node in self.ag.nodes:
            node_x, node_y = self.positions[node]
            x.append(node_x)
            y.append(node_y)

            # Build the hovertext
            node_hovertext = str("id: {}".format(node))
            if hasattr(self, "ranking_values"):
                node_hovertext += "<br>ranking position: {}/{}".format(
                    self.ranking_order[node], self.ag.number_of_nodes())
                node_hovertext += "<br>ranking value: {:0.2E}".format(
                    self.ranking_values[node])
            hovertext.append(node_hovertext)

        # Create the trace
        marker_size = 12
        self.node_trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            hoverinfo="text",
            hovertext=hovertext,
            hoverlabel=dict(font=dict(family="Montserrat")),
            marker=dict(size=marker_size, color=ui.constants.color_accent))

        # Update the color of the nodes if ranking has been applied
        if hasattr(self, "ranking_values"):
            self.node_trace.marker = dict(size=marker_size,
                                          showscale=False,
                                          colorscale="Reds",
                                          color=list(
                                              self.ranking_values.values()))

    def create_edge_trace(self):
        x = []
        y = []

        # Create the edges
        for edge in self.ag.edges:
            x0, y0 = self.positions[edge[0]]
            x1, y1 = self.positions[edge[1]]
            x += [x0, x1, None]
            y += [y0, y1, None]

        # Create the trace
        self.edge_trace = go.Scatter(x=x,
                                     y=y,
                                     mode="lines",
                                     line=dict(width=1.5,
                                               color=ui.constants.color_light))

    def create_cluster_trace(self):
        node_mapping = Spectral1(self.ag).apply(K=6)

        clusters = sorted(np.unique(list(node_mapping.values())))
        cluster_colors = dict([(cluster, create_random_color())
                               for cluster in clusters])

        cd = ClusterDrawer(self.ag, node_mapping)
        cd.apply()

        self.cluster_traces = []
        for zone in cd.zones:
            # Create the zone
            x = []
            y = []
            for line in zone["contour"]:
                start = line[0]
                end = line[1]
                x += [start[0], end[0], None]
                y += [start[1], end[1], None]

            # Create the trace
            color = cluster_colors[zone["cluster"]]
            self.cluster_traces.append(
                go.Scatter(x=x,
                           y=y,
                           mode="lines",
                           line=dict(width=4, color=color)))

    def create_graph(self) -> dcc.Graph:
        axis_parameters = dict(showgrid=False,
                               zeroline=False,
                               showticklabels=False)

        figure = go.Figure(
            data=[self.edge_trace, self.node_trace] + self.cluster_traces,
            layout=go.Layout(margin=dict(b=8, l=8, r=8, t=8),
                             showlegend=False,
                             paper_bgcolor=ui.constants.color_dark,
                             plot_bgcolor=ui.constants.color_dark,
                             xaxis=axis_parameters,
                             yaxis=axis_parameters))

        return dcc.Graph(id="graph",
                         figure=figure,
                         config=dict(displaylogo=False))
