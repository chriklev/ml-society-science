import random_banker
import group1_banker
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def get_data():
    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign', 'repaid']

    data_raw = pd.read_csv("../../data/credit/german.data",
                           delim_whitespace=True, names=features)
    numeric_variables = ['duration', 'age', 'residence time',
                         'installment', 'amount', 'persons', 'credits']
    data = pd.DataFrame(columns=features)
    data[numeric_variables] = data_raw[numeric_variables]

    # Mapping the response to 0 and 1
    data["repaid"] = data_raw["repaid"].map({1: 1, 2: 0})
    # Create dummy variables for all the catagorical variables
    not_dummy_names = numeric_variables + ["repaid"]
    dummy_names = [x not in not_dummy_names for x in features]
    dummies = pd.get_dummies(data_raw.iloc[:, dummy_names], drop_first=True)
    data = data.join(dummies)
    return data


def utility_from_obs(predicted_decision, true_decision, amount, duration, interest_rate):
    """Calculates utility from a single observation.

    Args:
        predicted_decision: the model's best action
        true_decision: if the observation repaid or not
        amount: the lending amount
        duration: the number of periods
        interest_rate: the interest rate of the loan

    Returns:
        The utility from the single observation given our action.
    """
    if predicted_decision == 1:
        if true_decision == 1:
            return amount*((1 + interest_rate)**duration - 1)
        else:
            return -amount
    else:
        return 0


def utility_from_test_set(X, y, decision_maker, interest_rate):
    """Calculates total utility from a given test set.

    Args:
        X: the covariates of the test set
        y: the response variable of the test set
        decision_maker: the decision maker to use in order to calculate utility
        interest_rate: the interest rate to use when calculating utility

    Returns:
        The sum of utility from the test set and the sum of utility divided by
        total amount.
    """

    num_obs = len(X)
    obs_utility = np.zeros(num_obs)
    obs_amount = np.zeros_like(obs_utility)

    for new_obs in range(num_obs):
        predicted_decision = decision_maker.get_best_action(X.iloc[new_obs])
        true_decision = y.iloc[new_obs]

        amount = X['amount'].iloc[new_obs]
        duration = X['duration'].iloc[new_obs]

        obs_utility[new_obs] = utility_from_obs(
            predicted_decision, true_decision, amount, duration, interest_rate)
        obs_amount[new_obs] = amount

    return np.sum(obs_utility), np.sum(obs_utility)/np.sum(obs_amount)


def compare_decision_makers(num_of_repeats, num_of_folds, response, interest_rate):
    """Tests the random banker against our group1 banker.

    Args:
        num_of_repeats: the number of tests to run
        response: the name of the response variable
        interest_rate: the interest rate to use when calculating utility
    """
    column_names = [
        "random_utility",
        "random_roi",
        "group1_utility",
        "group1_roi",
        "conservative_utility",
        "conservative_roi"
    ]
    results = pd.DataFrame(columns=column_names)

    # decision makers #
    # random banker
    r_banker = random_banker.RandomBanker()
    r_banker.set_interest_rate(interest_rate)

    # group1 banker
    n_banker = group1_banker.Group1Banker()
    n_banker.set_interest_rate(interest_rate)

    # conservative group1 banker
    c_banker = group1_banker.Group1Banker()
    c_banker.enable_utility_epsilon(max_alpha=0.05)
    c_banker.set_interest_rate(interest_rate)

    # get data
    data = get_data()
    # pop removes and returns the given column, "response" is no longer in data
    y = data.pop(response)

    # repeated cross validation
    i = 0
    for _ in range(num_of_repeats):
        kf = KFold(n_splits=num_of_folds, shuffle=True)
        for train_indices, test_indices in kf.split(data):
            X_train, X_test = \
                data.iloc[train_indices, :], data.iloc[test_indices, :]
            y_train, y_test = y[train_indices], y[test_indices]
            # fit models
            r_banker.fit(X_train, y_train)
            n_banker.fit(X_train, y_train)
            c_banker.fit(X_train, y_train)

            results.loc[i, "random_utility"], results.loc[i, "random_roi"] = utility_from_test_set(
                X_test, y_test, r_banker, interest_rate)
            results.loc[i, "group1_utility"], results.loc[i, "group1_roi"] = utility_from_test_set(
                X_test, y_test, n_banker, interest_rate)
            results.loc[i, "conservative_utility"], results.loc[i, "conservative_roi"] = utility_from_test_set(
                X_test, y_test, c_banker, interest_rate)
            i += 1
    return results


if __name__ == "__main__":
    import time
    t0 = time.time()
    np.random.seed(1)
    response = 'repaid'
    results = compare_decision_makers(50, 5, response, 0.1)
    print(time.time() - t0)

    print(results.describe())

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.distplot(results["random_utility"], label="Random banker")
    sns.distplot(results["group1_utility"], label="Group1 banker")
    sns.distplot(results["conservative_utility"], label="Conservative banker")
    plt.legend()
    plt.xlabel("Average utility over different random train/test draws")
    plt.ylabel("Density")
    plt.show()
