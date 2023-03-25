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
import pandas as pd
import numpy as np
import data

from python_ta.contracts import check_contracts


# @check_contracts
class GuessTree:
    """A binary tree class that stores the decision tree for a given dataset and algorithm.

        Representation Invariants:
            - (self.type == 'leaf') == (self.left is None)
            - (self.type == 'leaf') == (self.right is None)
            - (self.type == 'decision') == (self.value is None)
            - (self.type == 'decision') == (self.algorithm is not None)
            - (self.type == 'decision') == (self.feature_index is not None)
            - (self.type == 'decision') == (self.threshold is not None)
            - (self.type == 'decision') == (self.info_gain is not None)
            - self.info_gain is None or 0<= self.info_gain <= 1


        Instance Attributes:
            - node_type:
                The type of this node. Either 'leaf' or 'decision'.
            - left:
              The left subtree, or None if this tree is empty.
            - right:
              The right subtree, or None if this tree is empty.
            - algorithm:
                The algorithm used to build this tree.
            - feature_index:
                The index of the feature used to split this node.
            - threshold:
                 The threshold used to split this node.
            - info_gain:
                The information gain of this node.
            - value:
                The value of this node. Only used for leaf nodes.
          """
    node_type: str
    left: Optional[GuessTree]
    right: Optional[GuessTree]
    algorithm: Optional[str]

    # for decision node
    feature_index: Optional[int]
    threshold: Optional[float]
    info_gain: Optional[float]

    # for leaf node
    value: Optional[Any]

    def __init__(self, feature_index: Optional[int] = None, threshold: Optional[float] = None,
                 info_gain: Optional[float] = None, value: Optional[float] = None,
                 node_type: str = 'decision', algorithm: str = 'CART') -> None:
        """Initializes a new GuessTree.


        Preconditions:
            - node_type in {'decision', 'leaf'}
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'Multivariate'}
        """
        if node_type == 'leaf':
            self.left = None
            self.right = None
        else:
            self.left = GuessTree(None)
            self.right = GuessTree(None)

        self.algorithm = algorithm
        self.node_type = node_type

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.info_gain = info_gain

        # for leaf node
        self.value = value

    def __str__(self) -> str:
        """Return a string representation of this BST.

        This string uses indentation to show depth.

        """
        return self.str_indented(0)

    def str_indented(self, depth: int) -> str:
        """Return an indented string representation of this BST.

        The indentation level is specified by the <depth> parameter.

        Preconditions:
            - depth >= 0
        """

        if self.node_type == 'decision':
            return (depth * '  ' + f'{self.threshold}\n'
                    + self.left.str_indented(depth + 1)
                    + self.right.str_indented(depth + 1))
        else:
            return depth * '  ' + f'{self.value}\n'


if __name__ == '__main__':
    pass
    # import python_ta
    #
    # python_ta.check_all(config={
    #     'extra-imports': ['pandas', 'numpy', 'python_ta.contracts', 'typing', 'doctest'],
    #     'allowed-io': [],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200', 'R0913', 'W0622', 'R0902']
    # })
