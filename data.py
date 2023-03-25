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

from python_ta.contracts import check_contracts

# dataset for GuessWho game (24 characters) with 24 features (binary)
DATASET = pd.DataFrame({
    # hairstyle
    'hair_partition': [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    'curly_hair': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'hat': [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'bald': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    'long_hair': [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # hair color
    'ginger_hair': [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'white_hair': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    'brown_hair': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    'blond_hair': [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'black_hair': [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    # facial attributes
    'big_mouth': [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    'big_nose': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    'red_cheeks': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    'blue_eyes': [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    'sad_looking': [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    # facial hair
    'facial_hair': [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    'moustache': [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'beard': [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    # other
    'glasses': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    'earrings': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'female': [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # names
    'name': ['alex', 'alfred', 'anita', 'anne', 'bernard', 'bill', 'charles', 'claire',
             'david', 'eric', 'frans', 'george', 'herman', 'joe', 'maria', 'max', 'paul',
             'peter', 'philip', 'richard', 'robert', 'sam', 'susan', 'tom']
})

# features to use for the dataset
features = np.array([feature for feature in DATASET.columns if feature != 'name'])

# column description for the dataset
column_description = {
    "hair_partition": "Does your character have a visible hair partition?",
    "curly_hair": "Does your character have curly hair?",
    "hat": "Does your character wear a hat?",
    "bald": "Is your character bald?",
    "long_hair": "Does your character have long hair?",
    "ginger_hair": "Does your character have ginger hair?",
    "white_hair": "Does your character have white hair?",
    "brown_hair": "Does your character have brown hair?",
    "blond_hair": "Does your character have blond hair?",
    "black_hair": "Does your character have black hair?",
    "big_mouth": "Does your character have a big mouth?",
    "big_nose": "Does your character have a big nose?",
    "red_cheeks": "Does your character have red cheeks?",
    "blue_eyes": "Does your character have blue eyes?",
    "sad_looking": "Does your character look sad?",
    "facial_hair": "Does your character have facial hair?",
    "moustache": "Does your character have a moustache?",
    "beard": "Does your character have a beard?",
    "glasses": "Does your character wear glasses?",
    "earrings": "Does your character wear earrings?",
    "female": "Is your character female?"
}

if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['pandas', 'numpy', 'python_ta.contracts', 'typing', 'doctest'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200', 'R0913', 'W0622', 'R0902']
    })
