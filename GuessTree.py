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


@check_contracts
class GuessTree:
    """A binary tree class that stores the decision tree for a given dataset and algorithm.

        Representation Invariants:
            - (self._type == 'leaf') == (self._left is None)
            - (self._type == 'leaf') == (self._right is None)
            - (self._type == 'decision') == (self._value is None)
            - (self._type == 'decision') == (self._algorithm is not None)
            - (self._type == 'decision') == (self._feature is not None)
            - (self._type == 'decision') == (self._threshold is not None)
            - (self._type == 'decision') == (self._info_gain is not None)
          """
    # Private Instance Attributes:
    #     - _node_type: The type of this node. Either 'leaf' or 'decision'.
    #     - _left: The left subtree, or None if this tree is empty.
    #     - _right: The right subtree, or None if this tree is empty.
    #     - _algorithm: The algorithm used to build this tree.
    #     - _feature_ind: The index of the feature used to split this node.
    #     - _feature: The feature used to split this node.
    #     - _threshold: The threshold used to split this node.
    #     - _info_gain: The information gain of this node.
    #     - _value: The value of this node. Only used for leaf nodes.

    _node_type: str
    _left: Optional[GuessTree]
    _right: Optional[GuessTree]
    _algorithm: Optional[str]

    # decision node
    _feature_ind: Optional[int]
    _feature: Optional[str]
    _threshold: Optional[float]
    _info_gain: Optional[float]

    # leaf node
    _value: Optional[Any]

    def __init__(self, left: Optional[GuessTree] = None, right: Optional[GuessTree] = None,
                 feature_ind: Optional[int] = None, feature: Optional[str] = None,
                 threshold: Optional[float] = None, info_gain: Optional[float] = None, value: Optional[float] = None,
                 node_type: str = 'decision', algorithm: str = 'CART') -> None:
        """Initializes a new GuessTree.


        Preconditions:
            - node_type in {'decision', 'leaf'}
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'variance'}
        """
        if node_type == 'leaf':
            self._left = None
            self._right = None
        else:
            self._left = left
            self._right = right

        self._algorithm = algorithm
        self._node_type = node_type

        # for decision node
        self._feature_ind = feature_ind
        self._feature = feature
        self._threshold = threshold
        self._info_gain = info_gain

        # for leaf node
        self._value = value

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

        if self._node_type == 'decision':
            return (depth * '  ' + f'{self._feature} < {self._threshold} ig= {self._info_gain}  \n'
                    + self._left.str_indented(depth + 1)
                    + self._right.str_indented(depth + 1))
        else:
            return depth * '  ' + f'{self._value}\n'

    def write_guess_tree(self, filename: str) -> None:
        """Write this GuessTree to a file.

        Preconditions:
            - filename ends with '.pkl'
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def read_guess_tree(filename: str) -> GuessTree:
        """Read a GuessTree from a file and returns it.

        Preconditions:
            - filename ends with '.pkl'
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def get_guess(self) -> str:
        """Return the feature or value of this node."""
        return self._feature if self._feature is not None else self._value

    def traverse_tree(self, condition: bool) -> GuessTree:
        """Traverse the tree based on the condition corresponding to the feature of the current node
        and return the corresponding feature along with the subtree, or a leaf (and therefore a guess).

        Preconditions:
            - self._node_type == 'decision'
        """

        return self._right if condition else self._left

        # if condition:
        #     if self._right._node_type == 'leaf':
        #         return self._right._value
        #     else:
        #         return [self._right._feature, self._right]
        # else:
        #     if self._left._node_type == 'leaf':
        #         return self._left._value
        #     else:
        #         return [self._left._feature, self._left]

    def get_height(self) -> int:
        """Return the height of the tree."""
        if self._node_type == 'leaf':
            return 0
        else:
            return 1 + max(self._left.get_height(), self._right.get_height())

    def get_heights(self) -> list[int]:
        """Return a list of heights from the root of the tree to each leaf of the tree."""
        if self._node_type == 'leaf':
            return [0]
        else:
            heights_list = []
            left_heights = [1 + height for height in self._left.get_heights()]
            right_heights = [1 + height for height in self._right.get_heights()]
            heights_list.extend(left_heights)
            heights_list.extend(right_heights)
            return heights_list


@check_contracts
class DecisionTreeGenerator:
    """
    A class that generates a decision tree based on the algorithm specified.

    Representation Invariants:
        - self.min_splits >= 2
        - self.max_depth >= 0
        - gTree is a GuessTree object

    Instance Attributes:
        - min_splits: The minimum number of splits required to build the tree.
        - max_depth: The maximum depth of the tree.
    """
    # Private Instance Attributes:
    #     - _gTree: The GuessTree object that stores the decision tree.

    _gTree: Optional[GuessTree]
    min_splits: int
    max_depth: int

    def __init__(self, min_splits: Optional[int] = 2, max_depth: Optional[int] = 6) -> None:
        """Initializes a new DecisionTreeGenerator."""
        self._gTree = None
        self.min_splits = min_splits
        self.max_depth = max_depth

    def build_tree(self, dataset: np.ndarray, features: np.array, algorithm: str = 'CART',
                   curr_depth: int = 0) -> GuessTree:
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
                left_subtree = self.build_tree(best_split['left_data'], features, algorithm, curr_depth + 1)
                right_subtree = self.build_tree(best_split['right_data'], features, algorithm, curr_depth + 1)
                return GuessTree(left_subtree, right_subtree, best_split['feature_ind'],
                                 features[best_split['feature_ind']], best_split['threshold'],
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

    def split_dataset(self, dataset: np.ndarray, feature_ind: int, threshold: int) -> tuple[np.array, np.array]:
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

    def calculate_leaf_value(self, dataset: np.ndarray) -> str:
        """Returns the leaf value of the dataset."""
        return dataset[-1]

    def fit(self, dataset: pd.DataFrame, algorithm: str = 'CART') -> None:
        """Fits the decision tree to the dataset.

        Preconditions:
            - algorithm in {'CART', 'ID3', 'C4.5','Chi-squared', 'variance'}
        """
        x = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values.reshape(-1, 1)
        features = np.array(dataset.columns)
        new_dataset = np.concatenate((x, y), axis=1)
        self._gTree = self.build_tree(new_dataset, features, algorithm)

    def get_gametree(self) -> GuessTree:
        """Returns the GuessTree object."""
        return self._gTree


def tree_runner(file_name: str) -> list[GuessTree]:
    """Runs the decision tree generator for the given dataset and writes the tree to a file.
    """
    tree_list = []
    for algorithm in ['CART', 'ID3', 'C4.5', 'Chi-squared', 'variance']:
        dataset = pd.read_csv(file_name)
        generator = DecisionTreeGenerator(2, 15)
        generator.fit(dataset, algorithm)
        new_tree = generator.get_gametree()
        new_file = 'data/' + algorithm + '_tree.pkl'
        new_tree.write_guess_tree(new_file)
        tree_list.append(new_tree)
    return tree_list


if __name__ == '__main__':
    # tree_runner('data/guess_who.csv')
    pass
