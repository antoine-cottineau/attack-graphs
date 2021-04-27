import dash_core_components as dcc
import dash_html_components as html


def generate_section() -> html.Section:
    children = []

    children.append(html.Div(children="Load", className="section-subheader"))
    children.append(generate_subsection_load())

    children.append(
        html.Div(children="Generate", className="section-subheader"))
    children.append(generate_subsection_generate())

    children.append(html.Div(children="Save", className="section-subheader"))
    children.append(generate_subsection_save())

    return html.Section(id="section-attack-graphs",
                        children=children,
                        className="section")


def generate_subsection_load() -> html.Div:
    children = []

    children.append(
        dcc.Upload(id="graph-upload",
                   children=html.Div("Click or drag and drop a file")))

    return html.Div(children=children, className="subsection")


def generate_subsection_generate() -> html.Div:
    children = []

    slider_tooltip = dict(placement="right")

    children.append(
        html.Div(children="Total number of propositions",
                 className="method-tooltip"))
    children.append(
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

    children.append(
        html.Div(children="Number of initial propositions",
                 className="method-tooltip"))
    children.append(
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

    children.append(
        html.Div(children="Number of exploits", className="method-tooltip"))
    children.append(
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

    children.append(
        html.Button("Generate",
                    id="button-generate",
                    n_clicks=0,
                    className="section-button"))

    return html.Div(children=children, className="subsection")


def generate_subsection_save() -> html.Div:
    children = []

    children.append(
        dcc.Input(id="input-save",
                  placeholder="File location",
                  type="text",
                  className="input"))

    children.append(
        html.Button("Save",
                    id="button-save",
                    n_clicks=0,
                    className="section-button"))

    return html.Div(children=children, className="subsection")
