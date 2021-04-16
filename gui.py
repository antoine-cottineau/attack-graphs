import holoviews as hv
import networkx as nx
import panel as pn

from attack_graph import AttackGraph
from attack_graph_generation import Generator
from holoviews import opts
from pathlib import Path
from ranking.mehta import PageRankMethod, KuehlmannMethod

pn.extension(comms='vscode')


class UserInterface:
    def __init__(self):
        # Create a dummy graph for the beginning.
        self.ag = Generator().generate()

        self.create_layout()

    def show(self):
        self.base_layout.show()

    def create_layout(self):
        self.base_layout = pn.GridSpec(sizing_mode="stretch_both")

        # Add the two columns of the design to the base layout.
        self.left_column = pn.Column(sizing_mode="stretch_height")
        self.right_column = pn.Column(sizing_mode="stretch_height")

        self.base_layout[0, :2] = self.left_column
        self.base_layout[0, 2] = self.right_column

        # Fill the left column.
        self.update_left_column()

        # Fill the right column
        self.load_widget = self.create_load_widget()
        self.right_column.append(self.load_widget)

        self.generate_widget = self.create_generate_widget()
        self.right_column.append(self.generate_widget)

        self.ranking_widget = self.create_ranking_widget()
        self.right_column.append(self.ranking_widget)

    def update_left_column(self):
        self.graph = UserInterface.create_graph(self.ag)
        self.hierarchical_graph = UserInterface.create_hierarchical_graph()

        self.tabs = UserInterface.create_tabs(self.graph,
                                              self.hierarchical_graph)

        if len(self.left_column.objects) > 0:
            self.left_column.pop(0)

        self.left_column.append(self.tabs)

    @staticmethod
    def create_tabs(graph: pn.Row, hierarchical_graph: pn.Row) -> pn.Tabs:
        return pn.Tabs(("Graph", graph),
                       ("Hierarchical graph", hierarchical_graph))

    @staticmethod
    def create_graph(ag: AttackGraph) -> pn.Row:
        # To use the multipartite layout, the nodes must be given an attribute
        # called subset and corresponding to their layer.
        n_initial_propositions = len(ag.nodes[0]["ids_propositions"])
        for _, node in ag.nodes(data=True):
            node["subset"] = len(
                node["ids_propositions"]) - n_initial_propositions

        graph = hv.Graph.from_networkx(ag,
                                       nx.drawing.layout.multipartite_layout)

        has_ranking = "ranking_score" in ag.nodes[0]
        if has_ranking:
            options = opts.Graph(node_color="ranking_score",
                                 cmap="blues",
                                 colorbar=True)
        else:
            options = opts.Graph()
        graph.opts(options)

        return pn.panel(graph, sizing_mode="stretch_both")

    @staticmethod
    def create_hierarchical_graph() -> pn.Row:
        ag = AttackGraph()
        ag.load("graphs_input/AttackGraph.xml")

        graph = hv.Graph.from_networkx(ag, nx.drawing.layout.spring_layout)

        graph.opts(
            opts.Graph(arrowhead_length=0.007,
                       directed=True,
                       inspection_policy="edges"))

        return pn.panel(graph, sizing_mode="stretch_both")

    def create_load_widget(self) -> pn.WidgetBox:
        file_input = pn.widgets.FileInput(accept=".xml,.gml", margin=5)

        def on_click(self: UserInterface):
            if file_input.filename is None:
                # The user hasn't chosen a file.
                return

            # Decode the input file.
            decoded_string = file_input.value.decode("utf-8").replace("\n", "")

            # Create an attack graph with this file.
            self.ag = AttackGraph()
            self.ag.parse(decoded_string, Path(file_input.filename).suffix[1:])

            # Update the interface
            self.update_left_column()

        # Create a button that will trigger the loading of the attack graph.
        load_button = pn.widgets.Button(name="Load",
                                        button_type="primary",
                                        margin=5)
        load_button.on_click(lambda _: on_click(self))

        # Finalize the creation of the widget box
        load_box = pn.WidgetBox("### Load an attack graph",
                                file_input,
                                load_button,
                                sizing_mode="stretch_width")

        return load_box

    def create_generate_widget(self) -> pn.WidgetBox:
        # Create the sliders that handle the generation of attack graphs
        slider_n_propositions = pn.widgets.IntSlider(
            name="Total number of propositions",
            start=1,
            end=50,
            value=20,
            margin=5)
        slider_n_initial_propositions = pn.widgets.IntSlider(
            name="Initial number of true propositions",
            start=1,
            end=50,
            value=10,
            margin=5)
        slider_n_exploits = pn.widgets.IntSlider(name="Number of exploits",
                                                 start=1,
                                                 end=50,
                                                 value=20,
                                                 margin=5)

        # Create the button
        def on_click(self: UserInterface):
            n_propositions = slider_n_propositions.value
            n_initial_propositions = slider_n_initial_propositions.value
            n_exploits = slider_n_exploits.value

            self.ag = Generator(n_propositions, n_initial_propositions,
                                n_exploits).generate()

            self.update_left_column()

        generate_button = pn.widgets.Button(name="Generate",
                                            button_type="primary",
                                            margin=5)
        generate_button.on_click(lambda _: on_click(self))

        # Finalize the creation of the widget box
        generate_box = pn.WidgetBox("### Generate an attack graph",
                                    slider_n_propositions,
                                    slider_n_initial_propositions,
                                    slider_n_exploits,
                                    generate_button,
                                    sizing_mode="stretch_width")

        return generate_box

    def create_ranking_widget(self) -> pn.WidgetBox:
        method_selector = pn.widgets.Select(name="Method",
                                            options=["PageRank", "Kuehlmann"],
                                            margin=5)

        def on_click(self: UserInterface):
            if method_selector.value is None:
                return

            if method_selector.value == "PageRank":
                PageRankMethod(self.ag).apply()
            else:
                KuehlmannMethod(self.ag).apply()

            self.update_left_column()

        apply_button = pn.widgets.Button(name="Apply",
                                         button_type="primary",
                                         margin=5)
        apply_button.on_click(lambda _: on_click(self))

        ranking_box = pn.WidgetBox("### Apply a ranking algorithm",
                                   method_selector,
                                   apply_button,
                                   sizing_mode="stretch_width")

        return ranking_box


if __name__ == "__main__":
    ui = UserInterface()
    ui.show()
