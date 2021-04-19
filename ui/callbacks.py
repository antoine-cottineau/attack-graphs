import base64
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objects as go
import ui.constants
import ui.layout
import utils

from attack_graph import AttackGraph
from attack_graph_generation import Generator


def on_tab_selected(tab: str) -> html.Div:
    if tab == "side-menu-attack_graph":
        return ui.layout.generate_menu_attack_graphs()
    else:
        return html.Div()


def update_graph(context: object, data: list, filename: str,
                 n_propositions: int, n_initial_propositions: int,
                 n_exploits: int) -> str:
    if context.triggered[0]["prop_id"].split(".")[0] == "graph-upload":
        return on_attack_graph_loaded(data, filename)
    else:
        return on_button_generate_clicked(n_propositions,
                                          n_initial_propositions, n_exploits)


def on_attack_graph_loaded(data: str, filename: str) -> str:
    if data is None:
        return

    decoded_string = base64.b64decode(data.split(",")[1])
    extension = utils.get_file_extension(filename)

    ag = AttackGraph()
    ag.parse(decoded_string, extension)

    return ag.write()


def on_button_generate_clicked(n_propositions: int,
                               n_initial_propositions: int,
                               n_exploits: int) -> str:
    ag = Generator(n_propositions, n_initial_propositions,
                   n_exploits).generate()
    return ag.write()


def on_button_save_clicked(path: str, graph_json: str):
    if not path or not graph_json:
        return

    if utils.get_file_extension(path) != "json":
        return

    ag = AttackGraph()
    ag.parse(graph_json, "json")
    ag.save(path)


def on_attack_graph_changed(graph_json: str) -> dcc.Graph:
    if graph_json is None:
        return

    ag = AttackGraph()
    ag.parse(graph_json, "json")

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
        node_hovertext.append(str(node))

    node_trace = go.Scatter(x=node_x,
                            y=node_y,
                            mode="markers",
                            hoverinfo="text",
                            hovertext=node_hovertext,
                            marker=dict(size=10,
                                        color=ui.constants.color_accent))

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
