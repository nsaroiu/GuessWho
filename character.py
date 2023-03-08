import numpy as np


class Character():
    """A class representing a Guess Who character. Each character has a unique name and a unique set of identifiable
    features that players will use to identify their opponent's secret character.

    Representation Invariants:
        - name != ""

    """

    name: str
    features: np.ndarray

    def __init__(self, name: str, features: list[bool]) -> None:
        self.name = name
        self.features = np.array(features)
