import dash_core_components as dcc
import dash_html_components as html


def generate_section() -> html.Section:
    children = []

    children.append(
        html.Div(children="Select a method", className="method-tooltip"))

    children.append(
        dcc.Dropdown(id="dropdown-ranking",
                     options=[
                         dict(label="None", value="none"),
                         dict(label="PageRank", value="pagerank"),
                         dict(label="Kuehlmann", value="kuehlmann")
                     ],
                     value="none",
                     clearable=False,
                     searchable=False,
                     className="dropdown"))

    return html.Section(id="section-ranking",
                        children=children,
                        className="section")
