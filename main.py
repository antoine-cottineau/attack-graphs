from mulval import MulvalAttackGraph
from attack_graph import AttackGraph
from ranking import PageRankMethod, KuehlmannMethod
from graph_drawing import MulvalGraphDrawer, AttackGraphDrawer

# Create the attack graph
mag = MulvalAttackGraph()
mag.parse_from_file("./mulval_data/AttackGraph.xml")

# Draw the MulVAL attack graph
mgd = MulvalGraphDrawer(mag)
mulval_graph = mgd.create_pydot_graph("mulval_attack_graph")
MulvalGraphDrawer.save_graph_to_file(mulval_graph, "png")
simple_mulval_graph = mgd.create_pydot_graph("simple_mulval_attack_graph",
                                             simplified=True)
MulvalGraphDrawer.save_graph_to_file(simple_mulval_graph, "png")

# Convert to a standard attack graph
ag = AttackGraph()
ag.import_mulval_attack_graph(mag)

# Perform node ranking with PageRankMethod
prm = PageRankMethod(ag)
prm.apply()

# Draw the attack graph
agd = AttackGraphDrawer(ag)
attack_graph = agd.create_pydot_graph("ag_page_rank", labels=True)
AttackGraphDrawer.save_graph_to_file(attack_graph, "png")

# Perform node ranking with KuehlmannMethod
km = KuehlmannMethod(ag)
km.apply(max_m=7)

# Draw the attack graph
agd = AttackGraphDrawer(ag)
attack_graph = agd.create_pydot_graph("ag_kuehlmann", labels=True)
AttackGraphDrawer.save_graph_to_file(attack_graph, "png")
