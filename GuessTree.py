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


@check_contracts
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
            - node_type: The type of this node. Either 'leaf' or 'decision'.
            - left: The left subtree, or None if this tree is empty.
            - right: The right subtree, or None if this tree is empty.
            - algorithm: The algorithm used to build this tree.
            - feature_ind: The index of the feature used to split this node.
            - threshold: The threshold used to split this node.
            - info_gain: The information gain of this node.
            - value: The value of this node. Only used for leaf nodes.
          """
    node_type: str
    left: Optional[GuessTree]
    right: Optional[GuessTree]
    algorithm: Optional[str]

    # for decision node
    feature_ind: Optional[int]
    threshold: Optional[float]
    info_gain: Optional[float]

    # for leaf node
    value: Optional[Any]

    def __init__(self, left: Optional[GuessTree] = None, right: Optional[GuessTree] = None,
                 feature_ind: Optional[int] = None,
                 threshold: Optional[float] = None, info_gain: Optional[float] = None, value: Optional[float] = None,
                 node_type: str = 'decision', algorithm: str = 'CART') -> None:
        """Initializes a new GuessTree.


        Preconditions:
            - node_type in {'decision', 'leaf'}
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'variance'}
        """
        if node_type == 'leaf':
            self.left = None
            self.right = None
        else:
            self.left = left
            self.right = right

        self.algorithm = algorithm
        self.node_type = node_type

        # for decision node
        self.feature_ind = feature_ind
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
            return (depth * '  ' + f'{data.features[self.feature_ind]} < {self.threshold} ig= {self.info_gain}  \n'
                    + self.left.str_indented(depth + 1)
                    + self.right.str_indented(depth + 1))
        else:
            return depth * '  ' + f'{self.value}\n'


# @check_contracts
class DecisionTreeGenerator:
    """
    A class that generates a decision tree based on the algorithm specified.

    Representation Invariants:
        - min_splits >= 2
        - max_depth >= 0
        - gTree is a GuessTree object

    Instance Attributes:
        - gTree: The GuessTree object that stores the decision tree.
        - min_splits: The minimum number of splits required to build the tree.
        - max_depth: The maximum depth of the tree.
    """

    gTree: Optional[GuessTree]
    min_splits: int
    max_depth: int

    def __init__(self, min_splits: Optional[int] = 2, max_depth: Optional[int] = 6) -> None:
        """Initializes a new DecisionTreeGenerator."""
        self.min_splits = min_splits
        self.max_depth = max_depth

    def build_tree(self, dataset: np.ndarray, algorithm: str = 'CART', curr_depth: int = 0) -> GuessTree:
        """Builds a decision tree based on the algorithm specified.

        Preconditions:
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'variance'}
            - max_depth >= 0
            - min_splits >= 2
        """
        x, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(x)

        if num_samples >= self.min_splits and curr_depth < self.max_depth:
            best_split = self.get_best_split(dataset, num_features, algorithm)
            if best_split:
                left_subtree = self.build_tree(best_split['left_data'], algorithm, curr_depth + 1)
                right_subtree = self.build_tree(best_split['right_data'], algorithm, curr_depth + 1)
                return GuessTree(left_subtree, right_subtree, best_split['feature_ind'], best_split['threshold'],
                                 best_split['info_gain'], node_type='decision',
                                 algorithm=algorithm)

        leaf_value = self.calculate_leaf_value(y)
        return GuessTree(value=leaf_value, node_type='leaf')

    def get_best_split(self, dataset: np.ndarray, num_features: int, algorithm: str = 'CART') -> dict:
        """Returns the best split for the dataset based on the algorithm specified.

        Preconditions:
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'variance'}
        """
        best_split = {}
        max_info_gain = -np.inf

        for feature_ind in range(num_features):
            # feature_values = dataset[:, feature_ind]
            threshold = 1
            left_data, right_data = self.split_dataset(dataset, feature_ind, threshold)
            if len(left_data) > 0 and len(right_data) > 0:
                information_gain = self.get_information_gain(dataset, left_data, right_data, algorithm)
                if information_gain > max_info_gain:
                    max_info_gain = information_gain
                    best_split['feature_ind'] = feature_ind
                    best_split['threshold'] = threshold
                    best_split['left_data'] = left_data
                    best_split['right_data'] = right_data
                    best_split['info_gain'] = information_gain
        return best_split

    def split_dataset(self, dataset: np.ndarray, feature_ind: int, threshold: float) -> tuple[np.array, np.array]:
        """Splits the dataset based on the threshold specified."""
        left_data = dataset[dataset[:, feature_ind] < threshold]
        right_data = dataset[dataset[:, feature_ind] >= threshold]
        return left_data, right_data

    def get_information_gain(self, parent_data: np.ndarray, left_data: np.ndarray, right_data: np.ndarray,
                             algorithm: str) -> float:
        """Returns the information gain of the dataset based on the feature and threshold specified."""
        left_weight = len(left_data) / len(parent_data)
        right_weight = len(right_data) / len(parent_data)
        if algorithm == 'CART':
            information_gain = self.get_gini_index(parent_data) - left_weight * self.get_gini_index(
                left_data) - right_weight * self.get_gini_index(right_data)
        elif algorithm == 'ID3':
            information_gain = self.get_entropy(parent_data) - left_weight * self.get_entropy(
                left_data) - right_weight * self.get_entropy(right_data)
        elif algorithm == 'C4.5':
            information_gain = self.get_gain_ratio(parent_data, left_data, right_data)
        elif algorithm == 'Chi-squared':
            information_gain = self.get_chi_squared(parent_data, left_data, right_data)
        else:  # algorithm == 'variance'
            information_gain = self.get_variance(parent_data, left_data, right_data)
        return information_gain

    def get_entropy(self, dataset: np.ndarray) -> float:
        """Returns the entropy of the dataset."""
        num_samples = len(dataset)
        entropy = 0
        for label in np.unique(dataset[:, -1]):
            p = len(dataset[dataset[:, -1] == label]) / num_samples
            entropy += -p * np.log2(p)
        return entropy

    def get_gini_index(self, dataset: np.ndarray) -> float:
        """Returns the gini index of the dataset."""
        num_samples = len(dataset)
        gini_index = 0
        for label in np.unique(dataset[:, -1]):
            p = len(dataset[dataset[:, -1] == label]) / num_samples
            gini_index += p * (1 - p)
        return gini_index

    def get_gain_ratio(self, parent_data: np.ndarray, left_data: np.ndarray, right_data: np.ndarray) -> float:
        """Returns the gain ratio of the dataset based on the feature and threshold specified."""
        left_weight = len(left_data) / len(parent_data)
        right_weight = len(right_data) / len(parent_data)
        gain_info = self.get_entropy(parent_data) - left_weight * self.get_entropy(
            left_data) - right_weight * self.get_entropy(right_data)
        split_info = -left_weight * np.log2(left_weight) - right_weight * np.log2(right_weight)
        return gain_info / split_info

    def get_chi_squared(self, parent_data: np.ndarray, left_data: np.ndarray, right_data: np.ndarray) -> float:
        """Returns the chi-squared value of the dataset"""
        chi_squared = 0
        for label in np.unique(parent_data[:, -1]):
            p = len(parent_data[parent_data[:, -1] == label]) / len(parent_data)
            left_p = len(left_data[left_data[:, -1] == label]) / len(left_data)
            right_p = len(right_data[right_data[:, -1] == label]) / len(right_data)
            chi_squared += ((left_p - p) ** 2 / p + (right_p - p) ** 2 / p) ** 0.5
        return chi_squared

    def get_variance(self, parent_data: np.ndarray, left_data: np.ndarray, right_data: np.ndarray) -> float:
        """Returns the variance value of the dataset based on the feature and threshold specified."""
        left_weight = len(left_data) / len(parent_data)
        right_weight = len(right_data) / len(parent_data)
        left_variance = np.var(left_data[:, :-1])
        right_variance = np.var(right_data[:, :-1])
        parent_variance = np.var(parent_data[:, :-1])
        return parent_variance - left_weight * left_variance - right_weight * right_variance

    def calculate_leaf_value(self, dataset: np.ndarray) -> float:
        """Returns the leaf value of the dataset."""
        return dataset[-1]

    def fit(self, dataset: pd.DataFrame, algorithm: str = 'CART') -> None:
        """Fits the decision tree to the dataset.

        Preconditions:
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'variance'}
        """
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values.reshape(-1, 1)
        new_dataset = np.concatenate((x, y), axis=1)
        self.gTree = self.build_tree(new_dataset, algorithm)

    def get_gametree(self) -> GuessTree:
        """Returns the GuessTree object."""
        return self.gTree


if __name__ == '__main__':
    # dataset = data.DATASET
    # generator = DecisionTreeGenerator(2, 8)
    # generator.fit(dataset, 'multivariate')
    # guess_tree = generator.get_gametree()
    # print(guess_tree)

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['data', 'numpy', 'pandas', 'typing', 'guess_tree'],
        'max-line-length': 120,
        'disable': ['R1705', 'C0200']
    })
