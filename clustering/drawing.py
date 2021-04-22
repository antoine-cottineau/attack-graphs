from typing import Tuple
import networkx as nx
import numpy as np

from attack_graph import AttackGraph


class ClusterDrawer:
    def __init__(self, ag: AttackGraph, clusters: dict):
        self.ag = ag
        self.clusters = clusters

    def apply(self):
        self.create_layers()
        self.create_positions()
        self.create_node_dictionary()
        self.order_nodes_by_cluster()
        self.compute_node_distances()
        self.create_zones()
        self.add_zone_points()
        self.add_zone_contours()

    def create_layers(self):
        layers = dict()

        # Get the layer of each node
        n_initial_propositions = len(self.ag.nodes[0]["ids_propositions"])
        for id_node, ids_propositions in self.ag.nodes(
                data="ids_propositions"):
            layer = len(ids_propositions) - n_initial_propositions

            if layer in layers:
                current_nodes_in_layer = layers[layer]
                layers[layer] = current_nodes_in_layer + [id_node]
            else:
                layers[layer] = [id_node]

        self.layers = layers

    def create_positions(self):
        # Create a new graph not to change the existing one
        ag = self.ag.copy()

        # Add an attribute called "subset" to each one of the node
        for layer in list(self.layers.keys()):
            nodes_in_layer = self.layers[layer]
            for node in nodes_in_layer:
                ag.nodes[node]["subset"] = layer

        self.positions = nx.drawing.layout.multipartite_layout(ag)

    def create_node_dictionary(self):
        node_dictionary = dict()

        # Add each node to the dictionary with additional information (layer,
        # position and cluster)
        for layer in self.layers:
            nodes_in_layer = self.layers[layer]
            for node in nodes_in_layer:
                cluster = self.clusters[node]
                node_dictionary[node] = dict(layer=layer, cluster=cluster)

        self.node_dictionary = node_dictionary

    def order_nodes_by_cluster(self):
        new_layers = dict()
        for layer in self.layers:
            # Get information about the nodes in this layer
            nodes_in_layer = self.layers[layer]
            y_positions = sorted(
                [self.positions[node][1] for node in nodes_in_layer],
                reverse=True)
            clusters = [self.clusters[node] for node in nodes_in_layer]

            existing_clusters = sorted(np.unique(clusters))

            # Reorganize the nodes based on their respective cluster
            i_node = 0
            new_y_positions = dict()
            order = dict()
            for cluster in existing_clusters:
                for node in nodes_in_layer:
                    if self.node_dictionary[node]["cluster"] == cluster:
                        order[node] = i_node
                        new_y_positions[node] = y_positions[i_node]
                        i_node += 1

            # Update each layer so that the nodes are listed from top to bottom
            new_layer = np.zeros(len(nodes_in_layer), dtype=int)
            new_layer[list(order.values())] = list(order.keys())
            new_layers[layer] = list(new_layer)

            # Update positions and node_dictionary with the new values
            for node in nodes_in_layer:
                new_position = (self.positions[node][0], new_y_positions[node])
                self.positions[node] = new_position
                self.node_dictionary[node]["order"] = order[node]
                self.node_dictionary[node]["position"] = new_position

        self.layers = new_layers

    def compute_node_distances(self):
        # Compute the horizontal distance between two layers
        if len(self.layers.keys()) > 1:
            node_layer_0 = self.layers[0][0]
            node_layer_1 = self.layers[1][0]
            self.horizontal_distance = self.positions[node_layer_1][
                0] - self.positions[node_layer_0][0]
        else:
            # There is no real distance between two layers because there is
            # only one layer
            # In that case, we set the distance to 1
            self.horizontal_distance = 1

        # Compute the vertical distance between two adjacent nodes
        for nodes in self.layers.values():
            if len(nodes) > 1:
                self.vertical_distance = self.positions[
                    nodes[0]][1] - self.positions[nodes[1]][1]
                return

        # There is never two adjacent nodes in a same layer
        # In that case, we set the distance to 1
        self.vertical_distance = 1

    def create_zones(self):
        zones = []
        for layer in self.layers.keys():
            nodes_in_layer = self.layers[layer]

            if layer == 0:
                node = nodes_in_layer[0]
                zone = dict(cluster=self.clusters[node],
                            nodes=[node],
                            layers=[0])
                zones.append(zone)
                continue

            for node in nodes_in_layer:
                cluster = self.clusters[node]
                # Check if there is already a zone with the same cluster as
                # this node in the previous layer or in this layer
                matching_zone = [
                    (i_zone, zone) for i_zone, zone in enumerate(zones)
                    if zone["cluster"] == cluster and (
                        layer - 1 in zone["layers"] or layer in zone["layers"])
                ]

                if matching_zone:
                    # Update the zone and replace it in zones
                    i_zone, zone = matching_zone[0]
                    if layer not in zone["layers"]:
                        zone["layers"] = zone["layers"] + [layer]
                    zone["nodes"] = zone["nodes"] + [node]
                    zones[i_zone] = zone
                else:
                    # Create a new zone and add it to zones
                    zone = dict(cluster=cluster, nodes=[node], layers=[layer])
                    zones.append(zone)

        self.zones = zones

    def get_point(self, node: int, side: str) -> Tuple[float, float]:
        x, y = self.positions[node]
        if side == "left":
            return x - 0.2 * self.horizontal_distance, y
        elif side == "right":
            return x + 0.2 * self.horizontal_distance, y
        elif side == "top":
            return x, y + 0.3 * self.vertical_distance
        else:
            return x, y - 0.3 * self.vertical_distance

    def add_zone_points(self):
        for zone in self.zones:
            points = []
            nodes = zone["nodes"]

            for i, node in enumerate(nodes):
                layer = self.node_dictionary[node]["layer"]

                # Left point
                if layer - 1 not in zone["layers"]:
                    points.append((self.get_point(node, "left"), layer))

                # Right point
                if layer + 1 not in zone["layers"]:
                    points.append((self.get_point(node, "right"), layer))

                # Top point
                if i == 0 or i > 0 and self.node_dictionary[nodes[
                        i - 1]]["layer"] == layer - 1:
                    points.append((self.get_point(node, "top"), layer))

                # Bottom point
                if i == len(nodes) - 1 or i < len(
                        nodes) - 1 and self.node_dictionary[nodes[
                            i + 1]]["layer"] == layer + 1:
                    points.append((self.get_point(node, "bottom"), layer))

            zone["points"] = points

    def get_side_contour(self, points: list) -> list:
        _, list_y = zip(*points)
        order_y = np.flip(np.argsort(list_y))

        top = points[order_y[0]]
        bottom = points[order_y[-1]]

        if len(points) == 3:
            side = points[order_y[1]]
            return [(side, top), (side, bottom)]
        else:
            # We only need at most 4 points to draw this contour
            top_side = points[order_y[1]]
            bottom_side = points[order_y[-2]]
            return [(top, top_side), (top_side, bottom_side),
                    (bottom_side, bottom)]

    def add_zone_contours(self):
        for zone in self.zones:
            contour = []
            points_by_layer = [[
                point[0] for point in zone["points"] if point[1] == layer
            ] for layer in zone["layers"]]

            # Draw the side contours
            left_contour = self.get_side_contour(points_by_layer[0])
            right_contour = self.get_side_contour(points_by_layer[-1])

            contour += left_contour
            contour += right_contour

            # Draw the contours between the layers
            for i in range(len(points_by_layer) - 1):
                points = points_by_layer[i]
                points_next = points_by_layer[i + 1]

                top_left = max(points, key=lambda point: point[1])
                top_right = max(points_next, key=lambda point: point[1])
                bottom_left = min(points, key=lambda point: point[1])
                bottom_right = min(points_next, key=lambda point: point[1])

                contour += [(top_left, top_right), (bottom_left, bottom_right)]

            zone["contour"] = contour
