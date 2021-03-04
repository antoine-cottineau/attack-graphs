from mulval import MulvalAttackGraph
from ranking import PageRankMethod, KuehlmannMethod
from graph_drawing import GraphDrawer

# Create the attack graph
mag = MulvalAttackGraph()
mag.parse_from_file("./mulval_data/AttackGraph.xml")

# Perform node ranking with PageRankMethod
prm = PageRankMethod(mag)
prm.apply()

# Draw the attack graph for PageRankMethod
gd = GraphDrawer(mag)
graph = gd.create_pydot_graph("attack_graph_page_rank", simplified=False)
GraphDrawer.save_graph_to_file(graph, "dot")
GraphDrawer.save_graph_to_file(graph, "png")

# Perform node ranking with KuehlmannMethod
prm = KuehlmannMethod(mag)
prm.apply()

# Draw the attack graph for KuehlmannMethod
gd = GraphDrawer(mag)
graph = gd.create_pydot_graph("attack_graph_kuehlmann", simplified=False)
GraphDrawer.save_graph_to_file(graph, "dot")
GraphDrawer.save_graph_to_file(graph, "png")
