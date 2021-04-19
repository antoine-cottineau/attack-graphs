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


@app.callback(Output("attack-graph", "children"),
              Input("graph-upload", "contents"),
              Input("button-generate", "n_clicks"),
              State("graph-upload", "filename"),
              State("slider-n-propositions", "value"),
              State("slider-n-initial-propositions", "value"),
              State("slider-n-exploits", "value"))
def update_graph(data: list, n_clicks: int, filename: str, n_propositions: int,
                 n_initial_propositions: int, n_exploits: int) -> str:
    if n_clicks == 0:
        return
    context = dash.callback_context
    return ui.callbacks.update_graph(context, data, filename, n_propositions,
                                     n_initial_propositions, n_exploits)


@app.callback(Output("attack-graph", "className"),
              Input("button-save", "n_clicks"), State("input-save", "value"),
              State("attack-graph", "children"))
def on_button_save_clicked(n_clicks: int, path: str, graph_json: str):
    if n_clicks == 0:
        return
    ui.callbacks.on_button_save_clicked(path, graph_json)


@app.callback(Output("graph-zone", "children"),
              Input("attack-graph", "children"), State("graph-zone",
                                                       "children"))
def on_attack_graph_changed(graph_json: str,
                            current_graph: dcc.Graph) -> dcc.Graph:
    if graph_json is None:
        return current_graph
    return ui.callbacks.on_attack_graph_changed(graph_json)


if __name__ == "__main__":
    app.run_server(debug=True)
