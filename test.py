from attack_graph import AttackGraph
from clustering.drawing import ClusterDrawer

ag = AttackGraph()
ag.load("graphs_input/AttackGraph.xml")

cluster_mapping = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 0,
    5: 1,
    6: 2,
    7: 0,
    8: 2,
    9: 1,
    10: 0,
    11: 0,
    12: 1,
    13: 2
}

cd = ClusterDrawer(ag, cluster_mapping)
cd.apply()
