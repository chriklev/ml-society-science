""" Module for making data diferrentially private

# TODO: Figure out what to do with discrete numerical attributes
    Round them off after adding noise?
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
    new_values = rnd.choice(np.unique(data), size=transform_indexing.sum())
    new_data = data.copy()
    new_data[transform_indexing] = new_values
    return new_data


def transform_quantitative(data, b, scale_noise=False):
    """ Transform a column of quantitative data with laplace noise

    Args:
        data: Array with the data from a quantitative attribute.
        b: Positive float used as the second parameter of the laplace distribution,
            referred to as the scale or the diversity of the distribution.
        scale_noise: If true, scale the laplace noise by the standard deviation of the data.
            This allows the same value for b to be used on differently scaled data

    Returns:
        Array of the same length as the input data, containing the transformed data.
    """
    noise = rnd.laplace(0, b, size=data.size)
    if scale_noise:
        noise *= data.std()
    return data + noise


def apply_random_mechanism_to_data(data_frame, quantitative_names, categorical_names):
    """ Aplies a random mechanism to certain columns of a data frame

    Args:
        data_frame: A pandas data frame
        quantitative_names: An iterable with the column names of the quantitative attributes
            you wish to add laplace noise to.
        categorical_names: An iterable with the column names of the categorical attributes you
            wish to transform.

    Returns:
        Pandas data frame of the same dimentions as the one supplied, but with differentially private data.
    """
    dp_data = data_frame.copy()

    noise_b = 0.3
    for column_name in quantitative_names:
        dp_data[column_name] = transform_quantitative(
            data_frame[column_name], b=noise_b, scale_noise=True)

    p = 0.4
    for column_name in categorical_names:
        dp_data[column_name] = transform_categorical(
            data_frame[column_name], p)

    return dp_data


if __name__ == "__main__":
    """ Run this file to test how the module works on the german data.
    """
    import pandas as pd

    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign', 'repaid']

    data_raw = pd.read_csv("german.data",
                           delim_whitespace=True,
                           names=features)

    numeric_variables = set(['duration', 'age', 'residence time', 'installment',
                             'amount', 'persons', 'credits'])
    categorical_variables = set(features).difference(numeric_variables)

    print(data_raw.head())
    print(apply_random_mechanism_to_data(
        data_raw, numeric_variables, categorical_variables).head())
