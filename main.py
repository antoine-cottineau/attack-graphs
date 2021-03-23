from mulval import MulvalAttackGraph
from attack_graph import AttackGraph
from graph_drawing import AttackGraphDrawer
from deepwalk import DeepWalk

# Create the attack graph
mag = MulvalAttackGraph()
mag.parse_from_file("./mulval_data/AttackGraph.xml")

# Convert to a standard attack graph
ag = AttackGraph()
ag.import_mulval_attack_graph(mag)

# Perform clustering with DeepWalk
dw = DeepWalk(ag, 8, "deepwalk")
dw.run()
dw.cluster_with_k_clusters(3)

# Draw the attack graph
agd = AttackGraphDrawer(ag)
agd.create_pydot_graph("ag_deepwalk_clustering", labels=True)
agd.save_graph_to_file("png")
