import dash_core_components as dcc
import dash_html_components as html


def generate_layout() -> html.Div:
    return html.Div(id="root",
                    children=[
                        generate_section_attack_graph(),
                        generate_section_exploit_ranking(),
                        generate_section_exploit_selection(),
                        generate_section_clustering(),
                        html.Div(id="zone-attack-graph"),
                        dcc.Store(id="attack-graph"),
                        dcc.Store(id="parameters")
                    ])


def generate_section_attack_graph() -> html.Div:
    return html.Div(
        id="section-attack-graph",
        className="section",
        children=[
            html.H2(className="section-header", children="Attack graph"),
            html.Div(className="widget",
                     children=[
                         html.H3(className="widget-header",
                                 children="Load an attack graph"),
                         dcc.Upload(id="upload-attack-graph",
                                    children=html.Div("Click or drop a file"))
                     ]),
            html.Div(className="widget",
                     children=[
                         html.H3(className="widget-header",
                                 children="Generate an attack graph"),
                         dcc.RadioItems(
                             id="radio-items-graph-type",
                             options=[
                                 dict(label="State attack graph",
                                      value="state"),
                                 dict(label="Dependency attack graph",
                                      value="dependency")
                             ],
                             value="state"),
                         html.Div(id="sub-widget-number-exploits",
                                  children=[
                                      html.Div(children="Number of exploits"),
                                      dcc.Input(id="input-number-exploits",
                                                type="number",
                                                value=20)
                                  ]),
                         html.Button(id="button-generate", children="Generate")
                     ])
        ])


def generate_section_exploit_ranking() -> html.Div:
    return html.Div(
        id="section-exploit-ranking",
        className="section",
        children=[
            html.H2(className="section-header", children="Exploit ranking"),
            html.Div(className="widget",
                     children=[
                         html.H3(className="widget-header",
                                 children="Ranking method"),
                         dcc.Dropdown(id="dropdown-exploit-ranking-method",
                                      className="dropdown",
                                      options=[
                                          dict(label="None", value="none"),
                                          dict(label="PageRank",
                                               value="pagerank"),
                                          dict(label="Kuehlmann",
                                               value="kuehlmann"),
                                          dict(label="Value Iteration",
                                               value="vi"),
                                          dict(label="Homer", value="homer"),
                                          dict(label="Probabilistic path",
                                               value="pp")
                                      ],
                                      value="none",
                                      searchable=False,
                                      clearable=False)
                     ]),
            html.Div(className="table",
                     children=[
                         html.Div(className="table-header",
                                  children=[
                                      html.Div(className="table-cell",
                                               children=table_header)
                                      for table_header in
                                      ["#", "Exploit removed", "Score"]
                                  ]),
                         html.Div(id="table-exploit-ranking",
                                  className="table-content")
                     ])
        ])


def generate_section_exploit_selection() -> html.Div:
    return html.Div(id="section-exploit-selection",
                    className="section",
                    children=[
                        html.H2(className="section-header",
                                children="Exploit selection"),
                        dcc.Checklist(id="checklist-exploits",
                                      options=[],
                                      value=[])
                    ])


def generate_section_clustering() -> html.Div:
    return html.Div(
        id="section-clustering",
        className="section",
        children=[
            html.H2(className="section-header", children="Clustering"),
            html.Div(className="widget",
                     children=[
                         html.H3(className="widget-header",
                                 children="Clustering method"),
                         dcc.Dropdown(id="dropdown-clustering-method",
                                      className="dropdown",
                                      options=[
                                          dict(label="None", value="none"),
                                          dict(label="Spectral 1",
                                               value="spectral1"),
                                          dict(label="Spectral 2",
                                               value="spectral2"),
                                          dict(label="DeepWalk",
                                               value="deepwalk"),
                                          dict(label="GraphSAGE",
                                               value="graphsage"),
                                          dict(label="HOPE", value="hope")
                                      ],
                                      value="none",
                                      searchable=False,
                                      clearable=False)
                     ]),
            html.Div(className="table",
                     children=[
                         html.Div(className="table-header",
                                  children=[
                                      html.Div(className="table-cell",
                                               children=table_header)
                                      for table_header in
                                      ["Id", "Color", "Number of nodes"]
                                  ]),
                         html.Div(id="table-clustering",
                                  className="table-content")
                     ])
        ])
