from typing import Dict, List, Tuple


class ClusterDrawer:
    def __init__(self, positions: Dict[int, Tuple[float, float]],
                 clusters: Dict[str, dict]):
        # Each key of positions corresponds to a node id and each value
        # corresponds to a 2-dimensional point
        self.positions = positions
        # Each key of clusters corresponds to a cluster id and each value
        # corresponds to a dictionnary with two members: color and nodes
        self.clusters = clusters

        self.nodes = sorted(list(self.positions))
        self.node_assignment: Dict[int, str] = None
        self.layers: List[List[int]] = None
        self.horizontal_distance: float = None
        self.vertical_distance: float = None
        self.zones: List[dict] = None

    def apply(self):
        self.compute_node_assignment()
        self.compute_layers()
        self.reposition_nodes_by_cluster()
        self.compute_axis_distances()
        self.create_zones()
        self.add_zones_points()

    def compute_node_assignment(self):
        self.node_assignment = {}
        for i_cluster, data in self.clusters.items():
            nodes: List[int] = data["nodes"]
            for node in nodes:
                self.node_assignment[node] = i_cluster

    def compute_layers(self):
        layers: Dict[float, List[int]] = {}

        for node in self.nodes:
            x = self.positions[node][0]
            if x in layers:
                layers[x].append(node)
            else:
                layers[x] = [node]

        self.layers = []
        for x in sorted(list(layers)):
            self.layers.append(layers[x])

    def reposition_nodes_by_cluster(self):
        for i_layer, layer in enumerate(self.layers):
            # Compute the list of y of the nodes in the layer
            list_layer_y = []
            for node in layer:
                list_layer_y.append(self.positions[node][1])
            list_layer_y = sorted(list_layer_y, reverse=True)

            # Give a new position to each node in the layer based on its
            # cluster
            i_node = 0
            new_layer = []
            for data_cluster in self.clusters.values():
                cluster_nodes: List[int] = data_cluster["nodes"]
                for node in cluster_nodes:
                    if node in layer:
                        self.positions[node] = (self.positions[node][0],
                                                list_layer_y[i_node])
                        new_layer.append(node)
                        i_node += 1
            self.layers[i_layer] = new_layer

    def compute_axis_distances(self):
        # Compute the horizontal distance between two layers
        if len(self.layers) > 1:
            x0 = self.positions[self.layers[0][0]][0]
            x1 = self.positions[self.layers[1][0]][0]
            self.horizontal_distance = x1 - x0

        # Compute the vertical distance between two nodes in the same layer
        self.vertical_distance = float("inf")
        for layer in self.layers:
            if len(layer) == 1:
                continue
            y0 = self.positions[layer[0]][1]
            for node in layer:
                y1 = self.positions[node][1]
                distance = abs(y0 - y1)
                if distance != 0 and distance < self.vertical_distance:
                    self.vertical_distance = distance
            break

    def create_zones(self):
        self.zones = []
        for i_layer, layer in enumerate(self.layers):
            for node in layer:
                i_cluster = self.node_assignment[node]
                zone = self.get_zone(i_layer, i_cluster)
                if zone is None:
                    # Create a new zone
                    zone = dict(layers=[i_layer],
                                cluster=i_cluster,
                                nodes=[node])
                    self.zones.append(zone)
                else:
                    # Add the layer and the node to the zone
                    if i_layer not in zone["layers"]:
                        zone["layers"].append(i_layer)
                    zone["nodes"].append(node)

    def get_zone(self, i_layer, i_cluster) -> dict:
        for zone in self.zones:
            layers: List[int] = zone["layers"]
            if zone["cluster"] == i_cluster and (i_layer in layers
                                                 or i_layer - 1 in layers):
                return zone
        return None

    def add_zones_points(self):
        for zone in self.zones:
            points = []
            nodes: List[int] = zone["nodes"]
            layers: List[int] = zone["layers"]

            # Left points (from top to bottom)
            for layer_node in self.layers[layers[0]]:
                if layer_node in nodes:
                    points.append(self.get_point(layer_node, "left"))

            # Bottom points (from left to right)
            for i_layer in layers:
                for layer_node in reversed(self.layers[i_layer]):
                    if layer_node in nodes:
                        points.append(self.get_point(layer_node, "bottom"))
                        break

            # Right points (from bottom to top)
            for layer_node in reversed(self.layers[layers[-1]]):
                if layer_node in nodes:
                    points.append(self.get_point(layer_node, "right"))

            # Top points (from right to left)
            for i_layer in reversed(layers):
                for layer_node in self.layers[i_layer]:
                    if layer_node in nodes:
                        points.append(self.get_point(layer_node, "top"))
                        break

            zone["points"] = points

    def get_point(self, node: int, side: str) -> Tuple[float, float]:
        x, y = self.positions[node]
        horizontal_ratio = 0.2
        vertical_ratio = 0.3
        if side == "left":
            return x - horizontal_ratio * self.horizontal_distance, y
        elif side == "right":
            return x + horizontal_ratio * self.horizontal_distance, y
        elif side == "top":
            return x, y + vertical_ratio * self.vertical_distance
        else:
            return x, y - vertical_ratio * self.vertical_distance
