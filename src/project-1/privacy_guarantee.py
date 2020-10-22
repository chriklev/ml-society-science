import TestImplementation
import group1_banker
import differential_privacy
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def apply_epsilon_DP_noise(data, epsilon):
    dp_data = data.copy()
    numeric_variables = [
        'duration', 'age', 'residence time', 'installment',
        'amount', 'persons', 'credits'
    ]
    n_columns = len(dp_data.columns)
    col_epsilon = epsilon/n_columns

    for column in dp_data:
        if column in numeric_variables:
            val_range = dp_data[column].max() - dp_data[column].min()
            laplace_lambda = val_range/col_epsilon
            dp_data.loc[:, column] = differential_privacy.transform_quantitative(
                data=dp_data[column], b=laplace_lambda
            )
        else:
            rrm_p = 1/(np.exp(col_epsilon) + 1)
            dp_data.loc[:, column] = differential_privacy.transform_categorical(
                data=dp_data[column], p=rrm_p
            )

    return dp_data


def cv_error_epsilons(epsilon_sequence):
    """
    """
    banker = group1_banker.Group1Banker()
    banker.set_interest_rate(0.05)

    data = TestImplementation.get_raw_data()

    cv_errors = np.zeros_like(epsilon_sequence)
    n_folds = 5
    kf = KFold(n_splits=n_folds)
    for train, test in kf.split(data):

        X_train = data.iloc[train, :]
        X_train = TestImplementation.one_hot_encode(X_train)
        y_train = X_train.pop('repaid')

        for i, epsilon in enumerate(epsilon_sequence):
            X_test = data.iloc[test, :]
            y_test = X_test.pop('repaid').to_numpy()

            X_test = apply_epsilon_DP_noise(X_test, epsilon)
            X_test = TestImplementation.one_hot_encode(X_test)

            banker.fit(X_train, y_train)
            y_pred = banker.get_best_action(X_test)
            mse = np.mean((y_pred - y_test)**2)
            cv_errors[i] += mse/n_folds
        print("Done with a fold")

    return cv_errors


if __name__ == "__main__":
    epsilon_sequence = np.power(10, np.linspace(
        np.log10(0.001), np.log10(20), 300))
    cv_errors = cv_error_epsilons(epsilon_sequence)
    plt.scatter(np.log10(epsilon_sequence), cv_errors)
    plt.xlabel("log_10(epsilon)")
    plt.ylabel("Mean square test error")
    plt.savefig("img/privacy_guarantees")
    plt.show()
