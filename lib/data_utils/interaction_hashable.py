from typing import Tuple, Union


class CellInteraction:

    def __init__(self, type1: str, type2: str):

        self.type1 = type1
        self.type2 = type2

        self.interactions = tuple(sorted([type1, type2]))

    def torch_str_key(self):
        return self.interactions.__str__()

    def interaction_types(self) -> Tuple[str, str]:
        return self.interactions

    def __eq__(self, other: 'CellInteraction') -> bool:
        return self.interactions == other.interactions

    def __hash__(self):
        return self.interactions.__hash__()

    def clone(self) -> 'CellInteraction':
        return CellInteraction(self.type1, self.type2)


class FeatureInteraction(CellInteraction):

    def __init__(self, type1: str, type2: str, dist: float):
        super(FeatureInteraction, self).__init__(type1, type2)

        self.dist = dist

    def __str__(self):
        return 'FeatureInteraction({0})'.format(sorted(list(self.interactions)).__str__())

    def torch_str_key(self):
        return 'FeatureInteraction({0})'.format(sorted(list(self.interactions)).__str__())

    def deep_copy(self) -> 'FeatureInteraction':
        a_type, b_type = self.interaction_types()
        return FeatureInteraction(a_type, b_type, self.dist)
