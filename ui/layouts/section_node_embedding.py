import dash_core_components as dcc
import dash_html_components as html


def generate_section() -> html.Section:
    children = []

    children.append(
        html.Div(children="Select a method", className="method-tooltip"))

    children.append(
        dcc.Dropdown(id="dropdown-node-embedding",
                     options=[
                         dict(label="DeepWalk", value="deepwalk"),
                         dict(label="GraphSAGE", value="graphsage"),
                         dict(label="HOPE", value="hope")
                     ],
                     value="deepwalk",
                     clearable=False,
                     searchable=False,
                     className="dropdown"))

    children.append(
        html.Div(children="Enter a location", className="method-tooltip"))

    children.append(
        dcc.Input(id="input-node-embedding",
                  placeholder="File location",
                  type="text",
                  className="input"))

    children.append(
        html.Button("Apply",
                    id="button-apply-node-embedding",
                    n_clicks=0,
                    className="section-button"))

    return html.Section(id="section-node-embedding",
                        children=children,
                        className="section")
