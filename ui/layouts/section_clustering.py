import dash_core_components as dcc
import dash_html_components as html


def generate_section() -> html.Section:
    children = []

    children.append(
        html.Div(children="Select a method", className="method-tooltip"))

    children.append(
        dcc.Dropdown(id="dropdown-clustering",
                     options=[
                         dict(label="None", value="none"),
                         dict(label="Spectral 1", value="spectral1"),
                         dict(label="Spectral 2", value="spectral2"),
                         dict(label="DeepWalk", value="deepwalk"),
                         dict(label="GraphSAGE", value="graphsage"),
                         dict(label="HOPE", value="hope"),
                     ],
                     value="none",
                     clearable=False,
                     searchable=False,
                     className="dropdown"))

    return html.Section(id="section-clustering",
                        children=children,
                        className="section")
