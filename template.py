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

if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['pandas', 'numpy', 'python_ta.contracts', 'typing', 'doctest'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200', 'R0913', 'W0622', 'R0902']
    })
