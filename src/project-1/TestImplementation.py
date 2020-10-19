import random_banker
import group1_banker
import differential_privacy
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
    data = pd.DataFrame(columns=numeric_variables)
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


def repeated_cross_validation_utility(X, y, bankers, banker_names, interest_rate, n_repeats=20, n_folds=5):
    """ Preforms repeated cross validation to find estimates for average utility
    and return of investment for differnt bankers.

    Args:
        X: pandas data frame with covariates
        y: pandas series with the response
        bankers: iterable with bankers implementing the fit() and get_best_action() methods.
        banker_names: iterable with strings, containing the names of the bankers.
            Used to seperate the results in the "results" dictionary
        interest_rate: float interest rate by month
        n_repeats: number of repeats in repeated cross validation
        n_folds: number of folds in k-fold cross validation

    Returns:
        Dictionary on the form {string: numpy.ndarray(shape=(nrepeats, n_folds))}
    """
    results = {}
    for name in banker_names:
        results[name + "_utility"] = np.empty(shape=(n_repeats, n_folds))
        results[name + "_roi"] = np.empty(shape=(n_repeats, n_folds))

    for i in range(n_repeats):

        kf = KFold(n_splits=n_folds, shuffle=True)
        j = 0
        for train_indices, test_indices in kf.split(X):
            X_train = X.iloc[train_indices, :]
            X_test = X.iloc[test_indices, :]
            y_train = y[train_indices]
            y_test = y[test_indices]

            # fit models
            for banker in bankers:
                banker.fit(X_train, y_train)
            # find test scores
            for banker, name in zip(bankers, banker_names):
                util, roi = utility_from_test_set(
                    X_test, y_test, banker, interest_rate)
                results[name + "_utility"][i, j] = util
                results[name + "_roi"][i, j] = roi
            j += 1
    return results


def compare_decision_makers(n_repeats, n_folds, response, interest_rate):
    """Tests the random banker against our group1 banker.

    Args:
        num_of_repeats: the number of tests to run
        response: the name of the response variable
        interest_rate: the interest rate to use when calculating utility
    """

    ## decision makers ##
    # random banker
    r_banker = random_banker.RandomBanker()
    r_banker.set_interest_rate(interest_rate)

    # group1 banker
    g_banker = group1_banker.Group1Banker()
    g_banker.set_interest_rate(interest_rate)

    # conservative group1 banker
    c_banker = group1_banker.Group1Banker()
    c_banker.enable_utility_epsilon(max_alpha=0.1)
    c_banker.set_interest_rate(interest_rate)

    # get data
    data = get_data()
    # pop removes and returns the given column, "response" is no longer in data
    y = data.pop(response)

    return repeated_cross_validation_utility(
        X=data, y=y,
        bankers=[r_banker, g_banker, c_banker],
        banker_names=["random", "group1", "conservative"],
        interest_rate=interest_rate,
        n_repeats=n_repeats, n_folds=n_folds
    )


def get_differentially_private_data(laplace_lambda, p):
    """ Reads in the german data and applies a random mechanism

    Args:
        laplace_lambda: the lambda value to use in the laplace noise
        p: the probability of changing a categorical value

    Returns:
        Differentially private data set.
    """
    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign', 'repaid']

    data_raw = pd.read_csv("german.data",
                           delim_whitespace=True,
                           names=features)

    numeric_variables = ['duration', 'age', 'residence time', 'installment',
                         'amount', 'persons', 'credits']
    categorical_variables = set(features).difference(set(numeric_variables))

    data_raw = differential_privacy.apply_random_mechanism_to_data(
        data_raw, numeric_variables, categorical_variables, 0.3, 0.4)

    data = pd.DataFrame(columns=numeric_variables)
    data[numeric_variables] = data_raw[numeric_variables]

    # Mapping the response to 0 and 1
    data["repaid"] = data_raw["repaid"].map({1: 1, 2: 0})
    # Create dummy variables for all the catagorical variables
    not_dummy_names = numeric_variables + ["repaid"]
    dummy_names = [x not in not_dummy_names for x in features]
    dummies = pd.get_dummies(data_raw.iloc[:, dummy_names], drop_first=True)
    data = data.join(dummies)

    return data


def compare_preformance_differential_privacy(n_repeats, n_folds, response, interest_rate):
    """ Compare the perfomance of the model when using the original data vs using differentially private data

    Args:
        n_repeats: number of repeats in the repeated cross validation
        n_folds: number of folds in k-fold cv
        response: the name of the response variable
        interest_rate: the interest rate by month to use when calculating utility
    """
    g_banker = group1_banker.Group1Banker()
    g_banker.set_interest_rate(interest_rate)

    data = get_data()
    data_private = get_differentially_private_data(0.3, 0.4)

    y_normal = data.pop(response)
    y_private = data_private.pop(response)

    result_normal = repeated_cross_validation_utility(
        X=data, y=y_normal,
        bankers=[g_banker],
        banker_names=["normal_data"],
        interest_rate=interest_rate,
        n_repeats=n_repeats, n_folds=n_folds)

    result_private = repeated_cross_validation_utility(
        X=data_private, y=y_private,
        bankers=[g_banker],
        banker_names=["private_data"],
        interest_rate=interest_rate,
        n_repeats=n_repeats, n_folds=n_folds)

    return {**result_normal, **result_private}


def compare_privacy_garantees(laplace_lambdas, p, n_repeats, n_folds, response, interest_rate):
    """ Compare utility of models with differnt privacy guarantees.

    Args:
        laplace_lambdas: iterable with the different lambda values to use
        p: probability of changing a categorical variable
        n_repeats: number of repeats in the repeated cross validation
        n_folds: number of folds in k-fold cv
        response: the name of the response variable
        interest_rate: the interest rate by month to use when calculating utility

    Returns:
        Dictionary on the form {string: numpy.ndarray(shape=(nrepeats, n_folds))}
        with the results.
    """
    g_banker = group1_banker.Group1Banker()
    g_banker.set_interest_rate(interest_rate)

    data_frames = []
    data_frames.append(get_data())
    for laplace_lambda in laplace_lambdas:
        data_frames.append(get_differentially_private_data(laplace_lambda, p))

    results = {}
    for i, data_frame in enumerate(data_frames):
        y = data_frame.pop(response)

        new_result = repeated_cross_validation_utility(
            X=data_frame, y=y,
            bankers=[g_banker],
            banker_names=[f"lambda{i}"],
            interest_rate=interest_rate,
            n_repeats=n_repeats, n_folds=n_folds)

        print(f"Done with {i}/10")

        results.update(new_result)

    return results


if __name__ == "__main__":
    import time
    t0 = time.time()
    np.random.seed(1)
    response = 'repaid'
    """
    results = compare_decision_makers(
        n_repeats=20, n_folds=5, response=response, interest_rate=0.05)
    for key in results:
        results[key] = results[key].flatten()
    results = pd.DataFrame(results)
    print(results.describe())
    """
    """
    results_normal, results_private = compare_preformance_differential_privacy(
        n_repeats=50, n_folds=5, response=response, interest_rate=0.05
    )
    print(np.mean(results_private["private_data_utility"]) -
          np.mean(results_normal["normal_data_utility"]))
    """
    laplace_lambdas = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    results = compare_privacy_garantees(
        laplace_lambdas,
        0.4, 20, 5, response, 0.05)

    for key in results:
        results[key] = results[key].flatten()

    loss_in_utility = np.empty_like(laplace_lambdas)
    avg_utility_normal = np.mean(results["lambda0_utility"])
    i = 0
    for key in results:
        if "_utility" in key:
            if i != 0:
                loss_in_utility[i-1] = avg_utility_normal - \
                    np.mean(results[key])
            i += 1

    print(f"minutes elapsed: {(time.time() - t0)/60}")

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.plot(laplace_lambdas, loss_in_utility)
    plt.xlabel("lambda used in the laplace noise")
    plt.ylabel("difference (original - privatised)")
    plt.savefig("img/privacy_guarantees.png")
    plt.show()
    """
    sns.distplot(results_normal["normal_data_utility"], label="Original data")
    sns.distplot(results_private["private_data_utility"],
                 label="Differentially private data")
    plt.legend()
    plt.xlabel("Average utility over different random train/test draws")
    plt.ylabel("Density")
    plt.show()
    """
