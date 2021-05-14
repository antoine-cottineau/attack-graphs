import base64
import dash
import dash_core_components as dcc
import utils

from attack_graph import StateAttackGraph
from attack_graph_generation import Generator
from embedding.deepwalk import DeepWalk
from embedding.graphsage import GraphSage
from embedding.hope import Hope
from ui.graph_drawing import GraphDrawer
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

    ag = StateAttackGraph()
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

    ag = StateAttackGraph()
    ag.parse(graph_json, "json")
    ag.save(path)


def update_saved_parameters(ranking_method: str, clustering_method: str,
                            exploits: list, selected_exploits: list,
                            parameters: dict) -> dict:
    if parameters is None:
        new_parameters = dict()
    else:
        new_parameters = parameters.copy()

    if ranking_method:
        new_parameters["ranking_method"] = ranking_method

    if clustering_method:
        new_parameters["clustering_method"] = clustering_method

    if exploits:
        new_parameters["exploits"] = exploits

    if selected_exploits:
        new_parameters["selected_exploits"] = selected_exploits

    return new_parameters


def update_checklist_exploits(graph_json: str) -> Tuple[list, list]:

    if graph_json is None:
        return dash.no_update

    ag = StateAttackGraph()
    ag.parse(graph_json, "json")

    options = [
        dict(label=exploit[1], value=exploit[0])
        for exploit in ag.exploits.items()
    ]

    value = [*ag.exploits]

    return options, value


def update_section_visibility(current_visibility: str) -> Tuple[dict, str]:
    if current_visibility == "visibility":
        return dict(), "visibility_off"
    else:
        return dict(display="none"), "visibility"


def update_displayed_attack_graph(graph_json: str,
                                  parameters: dict) -> dcc.Graph:
    if graph_json is None:
        return dash.no_update

    ag = StateAttackGraph()
    ag.parse(graph_json, "json")

    return GraphDrawer(ag, parameters).draw()


def apply_node_embedding(method: str, path: str, graph_json: str):
    if graph_json is None or method is None or path is None:
        return dash.no_update

    ag = StateAttackGraph()
    ag.parse(graph_json, "json")

    if method == "deepwalk":
        embedding = DeepWalk(ag, 8)
    elif method == "graphsage":
        embedding = GraphSage(ag, 8, "ui")
    else:
        embedding = Hope(ag, 8, "cn")

    embedding.run()
    embedding.save_embedding_in_file(path)

    return dash.no_update
