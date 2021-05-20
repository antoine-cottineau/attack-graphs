import base64
import dash
import dash_core_components as dcc
import json
import utils
from attack_graph import BaseGraph, DependencyAttackGraph, StateAttackGraph
from attack_graph_generation import Generator
from ui.graph_drawing import DependencyAttackGraphDrawer, StateAttackGraphDrawer
from typing import Tuple


def update_saved_attack_graph(context: object, data: list, filename: str,
                              n_propositions: int, n_initial_propositions: int,
                              n_exploits: int, graph_type: str) -> str:
    if context.triggered[0]["prop_id"].split(".")[0] == "graph-upload":
        return load_attack_graph_from_file(data, filename)
    else:
        generator = Generator(n_propositions, n_initial_propositions,
                              n_exploits)
        if graph_type == "state":
            return generator.generate_state_attack_graph().write()
        else:
            return generator.generate_dependency_attack_graph().write()


def load_attack_graph_from_file(data: str, filename: str) -> str:
    if data is None:
        return

    decoded_string = base64.b64decode(data.split(",")[1])
    extension = utils.get_file_extension(filename)

    # Get the type of the attack graph
    type = json.loads(decoded_string)["type"]

    if type == "state":
        graph = StateAttackGraph()
    else:
        graph = DependencyAttackGraph()

    graph.parse(decoded_string, extension)
    return graph.write()


def save_attack_graph_to_file(path: str, graph_json: str):
    if not path or not graph_json:
        return

    if utils.get_file_extension(path) != "json":
        return

    # Get the type of the attack graph
    type = json.loads(graph_json)["type"]

    if type == "state":
        graph = StateAttackGraph()
    elif type == "dependency":
        graph = DependencyAttackGraph()

    graph.parse(graph_json, "json")
    graph.save(path)


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

    graph = BaseGraph()
    graph.parse(graph_json, "json")

    options = []
    for id_exploit, data in graph.exploits.items():
        label = data["text"]
        if len(label) > 100:
            label = label[:100] + "..."
        options.append(dict(value=id_exploit, label=label))

    value = list(graph.exploits)

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

    # Get the type of the attack graph
    type = json.loads(graph_json)["type"]

    if type == "state":
        graph = StateAttackGraph()
        graph.parse(graph_json, "json")
        return StateAttackGraphDrawer(graph, parameters).apply()
    elif type == "dependency":
        graph = DependencyAttackGraph()
        graph.parse(graph_json, "json")
        return DependencyAttackGraphDrawer(graph, parameters).apply()
