"""Copyright and Usage Information
===============================
Copyright (c) 2023 Prashanth Shyamala and Nicholas Saroiu
All Rights Reserved.
This program is the property of Prashanth Shyamala and Nicholas Saroiu.
All forms of distribution of this code, whether as given or with any changes,
are expressly prohibited.
For more information on copyright for CSC111 materials, please consult our Course Syllabus.
"""
from __future__ import annotations
from typing import Any, Optional
import pickle
import pandas as pd
import numpy as np

from python_ta.contracts import check_contracts


class Character:
    """A class representing a Guess Who character. Each character has a unique name and a unique set of identifiable
    features that players will use to identify their opponent's secret character.

    Representation Invariants:
        - self.name != ""
        - len(self._features) == 24

    Instance Attributes:
        - name: the name of the character

    """
    # Private Instance Attributes:
    #   - _features: a dictionary of the character's features, where the keys are the feature names
    #   and the values are booleans of wether the character has the feature or not

    name: str
    _features: dict[str, bool]

    def __init__(self, name: str, features: dict[str, bool]) -> None:
        self.name = name
        self._features = features

    def has_feature(self, feature: str) -> bool:
        """Return whether this character has the given feature.

        Preconditions:
            - feature in self._features
        """
        return self._features[feature]


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['numpy', 'pandas'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['E1136']
    })
