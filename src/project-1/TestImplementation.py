import random_banker
import name_banker
from sklearn.model_selection import train_test_split
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
    data = data_raw[numeric_variables]

    # Mapping the response to 0 and 1
    data["repaid"] = data_raw["repaid"].map({1: 1, 2: 0})
    # Create dummy variables for all the catagorical variables
    not_dummy_names = numeric_variables + ["repaid"]
    dummy_names = [x not in not_dummy_names for x in features]
    dummies = pd.get_dummies(data_raw.iloc[:, dummy_names], drop_first=True)
    data = data.join(dummies)
    return data


def utility_from_obs(predicted_decision, true_decision, amount, duration, interest_rate):
    if predicted_decision == 1:
        if true_decision == 1:
            return amount*((1 + interest_rate)**duration - 1)
        else:
            return -amount
    else:
        return 0


def utility_from_test_set(X, y, decision_maker, interest_rate):

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


def compare_decision_makers(num_of_tests, response, interest_rate):
    """Tests the random banker against our name banker.

    """
    bank_utility_random = np.zeros(num_of_tests)
    bank_investment_random = np.zeros_like(bank_utility_random)
    bank_utility_name = np.zeros(num_of_tests)
    bank_investment_name = np.zeros_like(bank_utility_name)

    # decision makers
    r_banker = random_banker.RandomBanker()
    r_banker.set_interest_rate(interest_rate)
    n_banker = name_banker.NameBanker()
    n_banker.set_interest_rate(interest_rate)

    # get data
    X = get_data()
    covariates = X.columns[X.columns != response]

    for i in range(num_of_tests):
        X_train, X_test, y_train, y_test = train_test_split(
            X[covariates], X[response], test_size=0.2)

        # fit models
        r_banker.fit(X_train, y_train)
        n_banker.fit(X_train, y_train)

        bank_utility_random[i], bank_investment_random[i] = utility_from_test_set(
            X_test, y_test, r_banker, interest_rate)
        bank_utility_name[i], bank_investment_name[i] = utility_from_test_set(
            X_test, y_test, n_banker, interest_rate)

    print(
        f"Avg. utility [random]\t= {np.sum(bank_utility_random)/num_of_tests}")
    print(
        f"Avg. ROI [random]    \t= {np.sum(bank_investment_random)/num_of_tests}")
    print(
        f"Avg. utility [name]  \t= {np.sum(bank_utility_name)/num_of_tests}")
    print(
        f"Avg. ROI [name]      \t= {np.sum(bank_investment_name)/num_of_tests}")


if __name__ == "__main__":
    response = 'repaid'

    compare_decision_makers(100, response, 0.05)
