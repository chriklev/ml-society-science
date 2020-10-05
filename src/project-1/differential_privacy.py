""" Module for transforming data to differentially private versions
"""

import numpy as np
import numpy.random as rnd


def transform_categorical(data, p):
    """ Transform a collumn of categorical data data with a randomised response mechanism

    Args:
        data: Nx1 ndarray with data from a categorical attribute
        p: The probablity of changing a datapoint

    Returns:
        Nx1 ndarray with the transformed data
    """
    transform_indexing = rnd.choice([True, False], size=data.size, p=[p, 1-p])
    # Draw new values uniformly for all datapoints that needs to be changed
    new_values = rnd.choice(np.unique(data), size=transform_indexing.sum())
    new_data = data.copy()
    new_data[transform_indexing] = new_values
    return new_data


if __name__ == "__main__":
    data = rnd.choice([0, 1, 2, 3], size=15)
    print(f"original data: {data}")
    print(f"new data:      {transform_categorical(data, 0.4)}")
