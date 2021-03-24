from mulval import MulvalAttackGraph
from attack_graph import AttackGraph
from graph_drawing import AttackGraphDrawer
from hope import Hope

# Create the attack graph
mag = MulvalAttackGraph()
mag.parse_from_file("./mulval_data/AttackGraph.xml")

# Convert to a standard attack graph
ag = AttackGraph()
ag.import_mulval_attack_graph(mag)

# Perform clustering with HOPE
hope = Hope(ag, 8, "cn")
hope.run()
hope.cluster_with_k_clusters(3)

# Draw the attack graph
agd = AttackGraphDrawer(ag)
agd.create_pydot_graph("ag_hope_clustering", labels=True)
agd.save_graph_to_file("png")
