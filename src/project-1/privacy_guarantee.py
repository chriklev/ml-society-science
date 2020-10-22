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

    utilities = np.zeros_like(epsilon_sequence)
    n_folds = 5
    kf = KFold(n_splits=n_folds)
    i_fold = 0
    for train, test in kf.split(data):
        i_fold += 1
        print(f"Started on fold {i_fold}/{n_folds}")

        X_train = data.iloc[train, :]
        X_train = TestImplementation.one_hot_encode(X_train)
        y_train = X_train.pop('repaid')

        for i, epsilon in enumerate(epsilon_sequence):
            X_test = data.iloc[test, :]
            y_test = X_test.pop('repaid').to_numpy()

            # We need to use original amount and durantions for calculating utility
            amount = X_test["amount"]
            duration = X_test["duration"]

            X_test = apply_epsilon_DP_noise(X_test, epsilon)
            X_test = TestImplementation.one_hot_encode(X_test)

            banker.fit(X_train, y_train)
            pred_decision = banker.get_best_action(X_test)
            utility = TestImplementation.utility_from_obs(
                pred_decision, y_test, amount, duration, 0.05)
            utilities[i] += np.sum(utility)/n_folds

    return utilities


if __name__ == "__main__":
    epsilon_sequence = np.linspace(1, 400, 200)
    cv_errors = cv_error_epsilons(epsilon_sequence)
    plt.scatter(epsilon_sequence/24, cv_errors)
    plt.xlabel("epsilon/k")
    plt.ylabel("Total utility")
    plt.savefig("img/privacy_guarantees_notlog.png")
    plt.show()
