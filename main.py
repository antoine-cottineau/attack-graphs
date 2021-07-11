from embedding.embedding import EmbeddingMethod
import dash
import dash_html_components as html
import json
import utils
from attack_graph import BaseGraph, DependencyAttackGraph, StateAttackGraph
from base64 import b64decode
from clustering.white_smyth import Spectral1, Spectral2
from dash.dependencies import Input, Output, State
from embedding.deepwalk import DeepWalk
from embedding.graphsage import GraphSage
from embedding.hope import Hope
from generation import Generator
from ranking.abraham import ProbabilisticPath
from ranking.homer import RiskQuantifier
from ranking.mehta import PageRankMethod, KuehlmannMethod
from ranking.sheyner import ValueIteration
from typing import Dict, List, Tuple
from ui.layout import generate_layout

app = dash.Dash(__name__)
app.layout = generate_layout()


@app.callback(Output("attack-graph", "data"),
              Input("upload-attack-graph", "contents"),
              Input("button-generate", "n_clicks"),
              State("upload-attack-graph", "filename"),
              State("radio-items-graph-type", "value"),
              State("input-number-exploits", "value"))
def update_attack_graph(graph_data: str, _: int, filename: str,
                        graph_type: str, n_exploits: int) -> str:
    if graph_data is None:
        # The user wants to generate an attack graph
        generator = Generator(n_exploits=n_exploits)
        if graph_type == "state":
            attack_graph = generator.generate_state_attack_graph()
        else:
            attack_graph = generator.generate_dependency_attack_graph()
    else:
        # The user wants to load an existing attack graph
        decoded_string = b64decode(graph_data.split(",")[1])
        extension = utils.get_file_extension(filename)
        attack_graph = get_attack_graph_from_string(decoded_string, extension)

    if attack_graph is None:
        return ""
    else:
        return attack_graph.write()


@app.callback(Output("table-exploit-ranking", "children"),
              Input("attack-graph", "data"),
              Input("dropdown-exploit-ranking-method", "value"))
def update_exploit_ranking(graph_data: str,
                           exploit_ranking_method: str) -> List[html.Div]:
    # Get the current attack graph
    attack_graph = get_attack_graph_from_string(graph_data)
    if attack_graph is None:
        return None

    # Get an instance of the ranking method
    instance = None
    if isinstance(attack_graph, StateAttackGraph):
        if exploit_ranking_method == "pagerank":
            instance = PageRankMethod(attack_graph)
        elif exploit_ranking_method == "kuehlmann":
            instance = KuehlmannMethod(attack_graph)
        elif exploit_ranking_method == "pp":
            instance = ProbabilisticPath(attack_graph)
    elif isinstance(attack_graph, DependencyAttackGraph):
        if exploit_ranking_method == "homer":
            instance = RiskQuantifier(attack_graph)
    if exploit_ranking_method == "vi":
        instance = ValueIteration(attack_graph)

    # Apply the method
    if instance is None:
        return
    ranking, scores = instance.rank_exploits()

    # Update the UI
    return get_table_exploit_ranking(ranking, scores)


@app.callback(Output("checklist-exploits", "options"),
              Output("checklist-exploits", "value"),
              Input("attack-graph", "data"))
def update_exploits(graph_data: str) -> Tuple[List[Dict[str, str]], List[str]]:
    # Get the current attack graph
    attack_graph = get_attack_graph_from_string(graph_data)
    if attack_graph is None:
        return None

    # Create the list of exploits
    exploits = []
    selected_exploits = []
    for id_exploit, data in attack_graph.exploits.items():
        text: str = data["text"]
        # Only keep the first 20 words
        text = " ".join(text.split(" ")[:20])
        # Add the id at the beginning of the text
        text = "{}: {}".format(id_exploit, text)
        exploits.append(dict(label=text, value=id_exploit))
        selected_exploits.append(id_exploit)

    return exploits, selected_exploits


@app.callback(Output("table-clustering", "children"),
              Output("parameters", "data"), Input("attack-graph", "data"),
              Input("dropdown-clustering-method", "value"),
              Input("checklist-exploits", "value"))
def update_clusters_and_parameters(
        graph_data: str, clustering_method: str,
        selected_exploits: List[str]) -> Tuple[List[html.Div], dict]:
    # Get the current attack graph
    attack_graph = get_attack_graph_from_string(graph_data)
    if attack_graph is None:
        return None

    # Remove the exploits that are not selected
    pruned_graph = attack_graph.get_pruned_graph(
        [int(i) for i in selected_exploits])

    # Get the list of clusters
    clusters = get_clusters(pruned_graph, clustering_method)

    # Create the table of clusters
    table_clustering = []
    if clusters is not None:
        table_clustering = []
        for id_cluster, data in clusters.items():
            table_clustering += [
                html.Div(className="table-cell", children=str(id_cluster)),
                html.Div(
                    className="table-cell",
                    style=dict(backgroundColor="{}".format(data["color"]))),
                html.Div(className="table-cell",
                         children=str(len(data["nodes"])))
            ]

    # Create the dictionary of parameters
    parameters = {}
    parameters["selected_exploits"] = selected_exploits
    parameters["clusters"] = clusters

    return table_clustering, parameters


def get_table_exploit_ranking(ranking: Dict[int, int],
                              scores: Dict[int, float]) -> List[html.Div]:
    table = []
    i_line = 0
    while len(table) // 3 < len(ranking):
        for id_exploit, position in ranking.items():
            if position == i_line:
                score = "{:.2e}".format(scores[id_exploit])
                if id_exploit is None:
                    id_exploit = "None"
                table += [position, id_exploit, score]
                break
        i_line += 1

    return [
        html.Div(className="table-cell", children=table[i])
        for i in range(len(table))
    ]


def get_attack_graph_from_string(string: str,
                                 extension: str = "json") -> BaseGraph:
    if string is None:
        return None

    # Get the type of the current attack graph
    data = json.loads(string)
    graph_type = data["type"]

    # Parse the data
    if graph_type == "state":
        attack_graph = StateAttackGraph()
    else:
        attack_graph = DependencyAttackGraph()
    attack_graph.parse(string, extension)

    return attack_graph


def get_clusters(attack_graph: BaseGraph, clustering_method: str) -> dict:
    # Get an instance of the clustering method
    instance = None
    if clustering_method == "spectral1":
        instance = Spectral1(attack_graph)
    elif clustering_method == "spectral2":
        instance = Spectral2(attack_graph)
    elif clustering_method == "deepwalk":
        instance = DeepWalk(attack_graph)
    elif clustering_method == "graphsage":
        instance = GraphSage(attack_graph)
    elif clustering_method == "hope":
        instance = Hope(attack_graph)

    if instance is None:
        return None
    elif isinstance(instance, EmbeddingMethod):
        instance.embed()

    # Apply clustering
    instance.cluster()

    # Create the result dictionary
    results = {}
    clusters = instance.clusters
    for id_cluster, nodes in clusters.items():
        color = utils.create_random_color()
        results[str(id_cluster)] = dict(color=color, nodes=nodes)

    return results


if __name__ == "__main__":
    app.run_server(debug=True)
