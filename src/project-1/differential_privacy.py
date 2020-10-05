""" Module for transforming data to differentially private versions
"""

import numpy as np
import numpy.random as rnd


def transform_categorical(data, p):
    """ Transform a column of categorical data data with a randomised response mechanism

    Args:
        data: Array with data from a categorical attribute.
        p: The probablity of changing a datapoint.

    Returns:
        Array of the same length as the input data, containing the transformed data.
    """
    transform_indexing = rnd.choice([True, False], size=data.size, p=[p, 1-p])
    # Draw new values uniformly for all datapoints that needs to be changed
    new_values = rnd.choice(np.unique(data), size=transform_indexing.sum())
    new_data = data.copy()
    new_data[transform_indexing] = new_values
    return new_data


def transform_quantitative(data, b, scale_data=False):
    """ Transform a column of quantitative data with laplace noise

    Args:
        data: Array with the data from a quantitative attribute.
        b: Positive float used as the second parameter of the laplace distribution,
            referred to as the scale or the diversity of the distribution.
        scale_data: If true, standardises the data before the noise is added, then
            reverts it back to the original scale.

    Returns:
        Array of the same length as the input data, containing the transformed data.
    """
    noise = rnd.laplace(0, b, size=data.size)
    if scale_data:
        mean = data.mean()
        std = data.std()
        data_scaled = (data - mean)/std
        return (data_scaled + noise)*std + mean
    return data + noise


if __name__ == "__main__":
    data = rnd.choice([0, 1, 2, 3], size=15)
    print(f"original data: {data}")
    print(f"new data:      {transform_categorical(data, 0.4)}")
