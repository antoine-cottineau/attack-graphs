import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.H1 import H1


def generate_layout() -> html.Div:
    return html.Div(id="root",
                    children=[
                        html.Div(id="attack-graph", style={"display": "none"}),
                        html.Div(id="dashboard-title",
                                 children=html.H1("Dashboard")),
                        generate_side_menu(),
                        html.Div(id="tools-menu"),
                        html.Div(id="graph-zone")
                    ])


def generate_side_menu() -> dcc.Tabs:
    side_menu_items = [{
        "id": "side-menu-attack_graph",
        "title": "Attack Graphs"
    }, {
        "id": "side-menu-ranking",
        "title": "Node ranking"
    }, {
        "id": "side-menu-clustering",
        "title": "Clustering"
    }, {
        "id": "side-menu-exploits",
        "title": "Exploits"
    }]

    tabs = [
        dcc.Tab(value=item["id"],
                label=item["title"],
                className="side-menu-tab",
                selected_className="selected") for item in side_menu_items
    ]

    return dcc.Tabs(id="side-menu-tabs",
                    children=tabs,
                    value=side_menu_items[0]["id"],
                    vertical=True)


def generate_menu_attack_graphs() -> html.Div:
    elements = []

    elements.append(
        html.H2(children="Load",
                className="tool-big-header",
                style=dict(marginTop="0")))
    elements.append(generate_menu_load())

    elements.append(html.H2(children="Generate", className="tool-big-header"))
    elements.append(generate_menu_generate())

    elements.append(html.H2(children="Save", className="tool-big-header"))
    elements.append(generate_menu_save())

    return html.Div(id="menu-attack-graphs", children=elements)


def generate_menu_load() -> html.Div:
    elements = []

    elements.append(
        dcc.Upload(id="graph-upload",
                   children=html.Div("Click or drag and drop a file")))

    return html.Div(children=elements, className="tool-sub-menu")


def generate_menu_generate() -> html.Div:
    elements = []

    slider_tooltip = dict(placement="right")

    elements.append(
        html.H3(children="Total number of propositions",
                className="tool-small-header"))
    elements.append(
        dcc.Slider(id="slider-n-propositions",
                   min=1,
                   max=50,
                   value=20,
                   marks={
                       1: "1",
                       50: "50"
                   },
                   tooltip=slider_tooltip,
                   className="slider"))

    elements.append(
        html.H3(children="Number of initial propositions",
                className="tool-small-header"))
    elements.append(
        dcc.Slider(id="slider-n-initial-propositions",
                   min=1,
                   max=50,
                   value=10,
                   marks={
                       1: "1",
                       50: "50"
                   },
                   tooltip=slider_tooltip,
                   className="slider"))

    elements.append(
        html.H3(children="Number of exploits", className="tool-small-header"))
    elements.append(
        dcc.Slider(id="slider-n-exploits",
                   min=1,
                   max=50,
                   value=20,
                   marks={
                       1: "1",
                       50: "50"
                   },
                   tooltip=slider_tooltip,
                   className="slider"))

    elements.append(
        html.Button("Generate",
                    id="button-generate",
                    n_clicks=0,
                    className="tool-button"))

    return html.Div(children=elements, className="tool-sub-menu")


def generate_menu_save() -> html.Div:
    elements = []

    elements.append(
        dcc.Input(id="input-save", placeholder="File location", type="text"))

    elements.append(
        html.Button("Save",
                    id="button-save",
                    n_clicks=0,
                    className="tool-button"))

    return html.Div(children=elements, className="tool-sub-menu")
