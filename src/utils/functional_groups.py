from typing import Any


class FunctionalGroup:
    def __init__(self, root_node: int,node_features: list[dict[Any, Any]],
                 edge_features: dict[tuple[int, int], dict[Any, Any]],
                 neighbours: list[list[int]], name: str) -> None:
        self.root_node = root_node
        self.node_features = node_features
        self.edge_features = edge_features
        self.neighbours = neighbours
        self.name = name

    def get_root_node_specs(self):
        return self.node_features[self.root_node]


class Ketone(FunctionalGroup):
    def __init__(self):
        node_features = [{"atomic_num": 8, "degree": 1}, {"atomic_num": 6, "degree": 3}]
        edge_features = {(0, 1): {"bond_type": "DOUBLE"}}
        neighbours = [[1], [0]]
        super().__init__(0, node_features, edge_features, neighbours, "Ketone (C=O)")


class Ether(FunctionalGroup):
    def __init__(self):
        node_features = [{"atomic_num": 8, "degree": 2}, {}, {}]
        edge_features = {(0, 1): {"bond_type": "SINGLE"}, (0, 2): {"bond_type": "SINGLE"}}
        neighbours = [[1, 2], [0], [0]]
        super().__init__(0, node_features, edge_features, neighbours, "Ether (R-O-R)")


class Alcohol(FunctionalGroup):
    def __init__(self):
        node_features = [{"atomic_num": 8, "degree": 1}, {}]
        edge_features = {(0, 1): {"bond_type": "SINGLE"}}
        neighbours = [[1], [0]]
        super().__init__(0, node_features, edge_features, neighbours, "Alcohol (R-OH)")


class Amine(FunctionalGroup):
    def __init__(self):
        node_features = [{"atomic_num": 7, "degree": 1}, {}]
        edge_features = {(0, 1): {"bond_type": "SINGLE"}}
        neighbours = [[1], [0]]
        super().__init__(0, node_features, edge_features, neighbours, "Amine (R-NH2)")
