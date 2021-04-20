import dash
import dash_core_components as dcc
import dash_html_components as html
import ui.callbacks
import ui.layout

from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = ui.layout.generate_layout()


@app.callback(Output("tools-menu", "children"), Input("side-menu-tabs",
                                                      "value"))
def on_tab_selected(tab: str) -> html.Div:
    return ui.callbacks.on_tab_selected(tab)


@app.callback(Output("attack-graph", "data"), Input("graph-upload",
                                                    "contents"),
              Input("button-generate", "n_clicks"),
              State("graph-upload", "filename"),
              State("slider-n-propositions", "value"),
              State("slider-n-initial-propositions", "value"),
              State("slider-n-exploits", "value"))
def update_saved_attack_graph(data: list, _: int, filename: str,
                              n_propositions: int, n_initial_propositions: int,
                              n_exploits: int) -> str:
    context = dash.callback_context
    return ui.callbacks.update_saved_attack_graph(context, data, filename,
                                                  n_propositions,
                                                  n_initial_propositions,
                                                  n_exploits)


@app.callback(Output("parameters", "data"), Input("dropdown-ranking", "value"),
              State("parameters", "data"))
def update_saved_parameters(ranking_method: str, parameters: dict) -> dict:
    return ui.callbacks.update_saved_parameters(parameters, ranking_method)


@app.callback(Output("useless-div", "children"),
              Input("button-save", "n_clicks"), State("input-save", "value"),
              State("attack-graph", "data"))
def save_attack_graph_to_file(_: int, path: str, graph_json: str):
    ui.callbacks.save_attack_graph_to_file(path, graph_json)
    return dash.no_update


@app.callback(Output("graph-zone", "children"), Input("attack-graph", "data"),
              Input("parameters", "data"))
def update_displayed_attack_graph(graph_json: str,
                                  parameters: dict) -> dcc.Graph:
    if graph_json is None:
        return dash.no_update

    return ui.callbacks.update_displayed_attack_graph(graph_json, parameters)


if __name__ == "__main__":
    app.run_server(debug=True)
