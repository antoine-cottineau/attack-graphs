import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import ranking.mehta as ranking
import ui.constants
import ui.layout
import utils

from attack_graph import AttackGraph
from attack_graph_generation import Generator
from typing import Tuple


def update_saved_attack_graph(context: object, data: list, filename: str,
                              n_propositions: int, n_initial_propositions: int,
                              n_exploits: int) -> str:
    if context.triggered[0]["prop_id"].split(".")[0] == "graph-upload":
        return load_attack_graph_from_file(data, filename)
    else:
        return generate_attack_graph(n_propositions, n_initial_propositions,
                                     n_exploits)


def load_attack_graph_from_file(data: str, filename: str) -> str:
    if data is None:
        return

    decoded_string = base64.b64decode(data.split(",")[1])
    extension = utils.get_file_extension(filename)

    ag = AttackGraph()
    ag.parse(decoded_string, extension)

    return ag.write()


def generate_attack_graph(n_propositions: int, n_initial_propositions: int,
                          n_exploits: int) -> str:
    ag = Generator(n_propositions, n_initial_propositions,
                   n_exploits).generate()
    return ag.write()


def save_attack_graph_to_file(path: str, graph_json: str):
    if not path or not graph_json:
        return

    if utils.get_file_extension(path) != "json":
        return

    ag = AttackGraph()
    ag.parse(graph_json, "json")
    ag.save(path)


def update_saved_parameters(ranking_method: str, exploits: list,
                            selected_exploits: list, parameters: dict) -> dict:
    if parameters is None:
        new_parameters = dict()
    else:
        new_parameters = parameters.copy()

    if ranking_method:
        new_parameters["ranking_method"] = ranking_method

    if exploits:
        new_parameters["exploits"] = exploits

    if selected_exploits:
        new_parameters["selected_exploits"] = selected_exploits

    return new_parameters


def update_checklist_exploits(graph_json: str) -> Tuple[list, list]:

    if graph_json is None:
        return dash.no_update

    ag = AttackGraph()
    ag.parse(graph_json, "json")

    options = [
        dict(label=exploit[1], value=exploit[0])
        for exploit in ag.exploits.items()
    ]

    value = [*ag.exploits]

    return options, value


def update_displayed_attack_graph(graph_json: str,
                                  parameters: dict) -> dcc.Graph:
    ag = AttackGraph()
    ag.parse(graph_json, "json")

    # Prune the graph if needed
    if parameters and "selected_exploits" in parameters:
        ids_exploits = [int(i) for i in parameters["selected_exploits"]]
        ag = ag.get_pruned_graph(ids_exploits)

    n = ag.number_of_nodes()

    # Apply ranking if needed
    ranking_values = None
    ranking_order = None
    if parameters and "ranking_method" in parameters and parameters[
            "ranking_method"] != "none":
        if parameters["ranking_method"] == "pagerank":
            ranking_values = ranking.PageRankMethod(ag).apply()
        elif parameters["ranking_method"] == "kuehlmann":
            ranking_values = ranking.KuehlmannMethod(ag).apply()

        # Compute the ranking order i.e. the position of each node in the
        # ranking
        ranking_order = np.zeros(n, dtype=int)
        indices_sorting = np.flip(np.argsort(list(ranking_values.values())))
        ranking_order[indices_sorting] = np.arange(n) + 1
        ranking_order = dict([(id, ranking_order[i])
                              for i, id in enumerate(ag.nodes)])

    # To use the multipartite layout, the nodes must be given an attribute
    # called subset and corresponding to their layer.
    n_initial_propositions = len(ag.nodes[0]["ids_propositions"])
    for _, node in ag.nodes(data=True):
        node["subset"] = len(node["ids_propositions"]) - n_initial_propositions

    positions = nx.drawing.layout.multipartite_layout(ag)

    # Build the graph
    edge_x = []
    edge_y = []
    for edge in ag.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x,
                            y=edge_y,
                            mode="lines",
                            line=dict(width=1.5,
                                      color=ui.constants.color_light))

    node_x = []
    node_y = []
    node_hovertext = []
    for node in ag.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        hovertext = str("id: {}".format(node))
        if ranking_values:
            hovertext += "<br>ranking position: {}/{}".format(
                ranking_order[node], n)
            hovertext += "<br>ranking value: {:0.2E}".format(
                ranking_values[node])
        node_hovertext.append(hovertext)

    node_trace = go.Scatter(x=node_x,
                            y=node_y,
                            mode="markers",
                            hoverinfo="text",
                            hovertext=node_hovertext,
                            hoverlabel=dict(font=dict(family="Montserrat")),
                            marker=dict(size=12,
                                        color=ui.constants.color_accent))

    # Add colors if ranking has been applied
    if ranking_values:
        node_trace.marker = dict(showscale=False,
                                 colorscale="RdPu",
                                 color=list(ranking_values.values()),
                                 size=10)

    figure = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(margin=dict(b=8, l=8, r=8, t=8),
                         showlegend=False,
                         paper_bgcolor=ui.constants.color_dark,
                         plot_bgcolor=ui.constants.color_dark,
                         xaxis=dict(showgrid=False,
                                    zeroline=False,
                                    showticklabels=False),
                         yaxis=dict(showgrid=False,
                                    zeroline=False,
                                    showticklabels=False)),
    )

    return dcc.Graph(id="graph", figure=figure, config=dict(displaylogo=False))
