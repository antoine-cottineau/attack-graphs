import dash_core_components as dcc
import dash_html_components as html

import ui.layouts.section_attack_graphs as section_attack_graphs
import ui.layouts.section_ranking as section_ranking
import ui.layouts.section_clustering as section_clustering
import ui.layouts.section_exploits as section_exploits


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

    children.append(
        html.H2(children="Attack graphs", className="section-header"))
    children.append(section_attack_graphs.generate_section())

    children.append(html.H2(children="Ranking", className="section-header"))
    children.append(section_ranking.generate_section())

    children.append(html.H2(children="Clustering", className="section-header"))
    children.append(section_clustering.generate_section())

    children.append(html.H2(children="Exploits", className="section-header"))
    children.append(section_exploits.generate_section())

    return html.Div(id="main-menu", children=children)
