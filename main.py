from mulval import MulvalAttackGraph
from attack_graph import AttackGraph
from ranking import PageRankMethod, KuehlmannMethod
from clustering import Spectral1, Spectral2
from graph_drawing import MulvalGraphDrawer, AttackGraphDrawer
from graphsage import Graphsage

# Create the attack graph
mag = MulvalAttackGraph()
mag.parse_from_file("./mulval_data/AttackGraph.xml")

# Draw the MulVAL attack graph
mgd = MulvalGraphDrawer(mag)
mgd.create_pydot_graph("mulval_attack_graph")
mgd.save_graph_to_file("png")
mgd.create_pydot_graph("simple_mulval_attack_graph", simplified=True)
mgd.save_graph_to_file("png")

# Convert to a standard attack graph
ag = AttackGraph()
ag.import_mulval_attack_graph(mag)

# Perform node ranking with PageRankMethod
prm = PageRankMethod(ag)
prm.apply()

# Perform clustering with Spectral1
spectral1 = Spectral1(ag)
spectral1.apply(5)

# Draw the attack graph
agd = AttackGraphDrawer(ag)
agd.create_pydot_graph("ag_page_rank_spectral_1", labels=True)
agd.save_graph_to_file("png")
agd.save_graph_to_file("dot")

# Perform node ranking with KuehlmannMethod
km = KuehlmannMethod(ag)
km.apply(max_m=7)

# Perform clustering with Spectral2
spectral2 = Spectral2(ag)
spectral2.apply(1, 5)

# Draw the attack graph
agd.create_pydot_graph("ag_kuehlmann_spectral_2", labels=True)
agd.save_graph_to_file("png")

# Perform clustering with Graphsage
graphsage = Graphsage(ag, 8, "graphsage")
graphsage.run()
graphsage.cluster_with_k_clusters(3)

# Draw the attack graph
agd.create_pydot_graph("ag_graphsage_clustering", labels=True)
agd.save_graph_to_file("png")
