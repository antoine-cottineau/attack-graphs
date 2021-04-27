import dash_core_components as dcc
import dash_html_components as html

import ui.layouts.section_attack_graphs as section_attack_graphs
import ui.layouts.section_ranking as section_ranking
import ui.layouts.section_clustering as section_clustering
import ui.layouts.section_exploits as section_exploits
import ui.layouts.section_node_embedding as section_node_embedding

section_ids = [
    "section-attack-graphs", "section-ranking", "section-clustering",
    "section-exploits", "section-node-embedding"
]
icon_ids = [
    "icon-attack-graph", "icon-ranking", "icon-clustering", "icon-exploits",
    "icon-node-embedding"
]
header_ids = [
    dict(section=section_ids[i], icon=icon_ids[i])
    for i in range(len(section_ids))
]


def generate_layout() -> html.Div:
    return html.Div(id="root",
                    children=[
                        dcc.Store(id="attack-graph"),
                        dcc.Store(id="parameters"),
                        html.Div(id="dashboard-title",
                                 children=html.H1("Dashboard")),
                        html.Div(id="graph-zone"),
                        generate_main_menu(),
                        html.Div(id="useless-div", style=dict(display="none"))
                    ])


def generate_main_menu() -> html.Div:
    children = []

    header_titles = [
        "Attack graphs", "Ranking", "Clustering", "Exploits", "Node embedding"
    ]
    header_visibilities = ["visibility"
                           ] + (len(header_titles) - 1) * ["visibility_off"]

    headers = [
        html.Div(children=[
            html.H2(children=header_titles[i],
                    className="section-header-label"),
            html.Button(id=icon_ids[i],
                        children=header_visibilities[i],
                        className="material-icons icon-visibility")
        ],
                 className="section-header") for i in range(len(section_ids))
    ]

    children.append(headers[0])
    children.append(section_attack_graphs.generate_section())

    children.append(headers[1])
    children.append(section_ranking.generate_section())

    children.append(headers[2])
    children.append(section_clustering.generate_section())

    children.append(headers[3])
    children.append(section_exploits.generate_section())

    children.append(headers[4])
    children.append(section_node_embedding.generate_section())

    return html.Div(id="main-menu", children=children)
