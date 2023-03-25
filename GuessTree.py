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
            - (self.type == 'decision') == (self.feature is not None)
            - (self.type == 'decision') == (self.threshold is not None)
            - (self.type == 'decision') == (self.info_gain is not None)


        Instance Attributes:
            - node_type:
                The type of this node. Either 'leaf' or 'decision'.
            - left:
              The left subtree, or None if this tree is empty.
            - right:
              The right subtree, or None if this tree is empty.
            - algorithm:
                The algorithm used to build this tree.
            - feature:
                The feature used to split this node.
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
    feature: Optional[str]
    threshold: Optional[float]
    info_gain: Optional[float]

    # for leaf node
    value: Optional[Any]

    def __init__(self, feature: Optional[str] = None, threshold: Optional[float] = None,
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
        self.feature = feature
        self.threshold = threshold
        self.info_gain = info_gain

        # for leaf node
        self.value = value

    def __str__(self) -> str:
        """Return a string representation of this GuessTree.

        This string uses indentation to show depth.

        """
        return self.str_indented(0)

    def str_indented(self, depth: int) -> str:
        """Return an indented string representation of this GuessTree.

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


class DecisionTreeGenerator:
    """
    A class that generates a decision tree based on the algorithm specified.

    Representation Invariants:
        - min_splits >= 2
        - max_depth >= 0
        - gTree is a GuessTree object

    Instance Attributes:
        - gTree:
            The GuessTree object that stores the decision tree.
        - min_splits:
            The minimum number of splits required to build the tree.
        - max_depth:
            The maximum depth of the tree.
    """

    gTree: Optional[GuessTree]
    min_splits: int
    max_depth: int

    def __int__(self, mins_splits: int = 2, max_depth: int = 6) -> None:
        """Initializes a new DecisionTreeGenerator."""
        self.min_splits = mins_splits
        self.max_depth = max_depth

    def build_tree(self, dataset: np.ndarray, algorithm: str = 'CART', curr_depth: int = 0) -> GuessTree:
        """Builds a decision tree based on the algorithm specified.

        Preconditions:
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'Multivariate'}
            - max_depth >= 0
            - min_splits >= 2
        """

    def get_best_split(self, dataset: np.ndarray, algorithm: str = 'CART') -> tuple:
        """Returns the best split for the dataset based on the algorithm specified.

        Preconditions:
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'Multivariate'}
        """

    def split_dataset(self, dataset: np.ndarray, feature: str, threshold: float) -> tuple:
        """Splits the dataset based on the feature and threshold specified."""

    def get_information_gain(self, parent_data: np.ndarray, feature: str, threshold: float) -> float:
        """Returns the information gain of the dataset based on the feature and threshold specified."""

    def get_entropy(self, dataset: np.ndarray) -> float:
        """Returns the entropy of the dataset."""

    def get_gini_index(self, dataset: np.ndarray) -> float:
        """Returns the gini index of the dataset."""

    def get_chi_squared(self, dataset: pd.DataFrame, feature: str, threshold: float) -> float:
        """Returns the chi squared value of the dataset based on the feature and threshold specified."""

    def get_multivariate(self, dataset: pd.DataFrame, feature: str, threshold: float) -> float:
        """Returns the multivariate value of the dataset based on the feature and threshold specified."""

    def calculate_leaf_value(self, dataset: pd.DataFrame) -> float:
        """Returns the leaf value of the dataset."""

    def fit(self, dataset: pd.DataFrame, algorithm: str = 'CART') -> None:
        """Fits the decision tree to the dataset.

        Preconditions:
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'Multivariate'}
        """
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values.reshape(-1, 1)
        new_dataset = np.concatenate((x, y), axis=1)
        self.gTree = self.build_tree(new_dataset, algorithm)

    def get_gametree(self) -> GuessTree:
        """Returns the GuessTree object."""
        return self.gTree


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
