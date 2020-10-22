import random_banker
import group1_banker
import differential_privacy
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf


def get_raw_data():
    """ Reads in raw data and only maps response to 0 and 1
    """
    features = ['checking account balance', 'duration', 'credit history',
                'purpose', 'amount', 'savings', 'employment', 'installment',
                'marital status', 'other debtors', 'residence time',
                'property', 'age', 'other installments', 'housing', 'credits',
                'job', 'persons', 'phone', 'foreign', 'repaid']

    data_raw = pd.read_csv("../../data/credit/german.data",
                           delim_whitespace=True, names=features)

    # Mapping the response to 0 and 1
    data_raw.loc[:, "repaid"] = data_raw["repaid"].map({1: 1, 2: 0})

    categorical_columns = ['checking account balance', 'credit history',
                           'purpose', 'savings', 'employment', 'marital status',
                           'other debtors', 'property', 'other installments',
                           'housing', 'job', 'phone', 'foreign', 'repaid']
    data_raw.loc[:, categorical_columns] = data_raw[categorical_columns].apply(
        lambda x: x.astype('category'))

    return data_raw


def one_hot_encode(data):
    """ One hot encodes specified columns.
    """
    columns = ['checking account balance', 'credit history',
               'purpose', 'savings', 'employment', 'marital status',
               'other debtors', 'property', 'other installments',
               'housing', 'job', 'phone', 'foreign']
    dummies = pd.get_dummies(data[columns], drop_first=True)
    data = data.drop(columns, axis=1)

    return data.join(dummies)


def get_data():
    data = get_raw_data()
    data = one_hot_encode(data)

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
    utility = np.zeros_like(true_decision)

    predicted_decision_bool = predicted_decision == 1
    ind1 = np.logical_and(predicted_decision_bool, true_decision == 1)
    ind2 = np.logical_and(predicted_decision_bool, true_decision == 0)

    utility[ind1] = amount[ind1]*((1 + interest_rate)**duration[ind1] - 1)
    utility[ind2] = -amount[ind2]

    return utility


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
    predicted_decision = decision_maker.get_best_action(X)

    amount = X['amount']
    duration = X['duration']

    utility = utility_from_obs(
        predicted_decision, y, amount, duration, interest_rate)

    return np.sum(utility), np.sum(utility)/np.sum(amount)


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


def _calculate_balance(df, threshold=None, upper=True):
    """Calculates the probability for the balance metric using relative
    frequency.

    Args:
        df: dataframe containing
            a: the action taken by the algorithm
            y: the true response
            z: the gender of the observation
            am: the amount of loan
        threshold: whether or not to threshold the amount
        upper: use upper part of threshold
    Returns:
        Calculated balance loss.
    """
    if (threshold == None):
        N = len(df)
        bal = 0
        for a in range(0, 2):
            for y in range(0, 2):
                p_y = len(df[df['y'] == y])/N
                p_a_and_y = len(df[(df['a'] == a) & (df['y'] == y)])/N
                p_a_y = p_a_and_y/p_y
                for z in range(0, 2):
                    p_y_z = len(df[(df['y'] == y) & (df['z'] == z)])/N
                    p_a_and_y_and_z = len(
                        df[(df['a'] == a) & (df['y'] == y) & (df['z'] == z)])/N
                    p_a_y_z = p_a_and_y_and_z/p_y_z
                    bal += (p_a_y_z - p_a_y)**2

        return bal
    elif upper:
        N = len(df['am'] > threshold)
        bal = 0
        for a in range(0, 2):
            for y in range(0, 2):
                p_y = len(df[(df['y'] == y) & (df['am'] > threshold)])/N
                p_a_and_y = len(df[(df['a'] == a) & (
                    df['y'] == y) & (df['am'] > threshold)])/N
                p_a_y = p_a_and_y/p_y
                for z in range(0, 2):
                    p_y_z = len(df[(df['y'] == y) & (
                        df['z'] == z) & (df['am'] > threshold)])/N
                    p_a_and_y_and_z = len(
                        df[(df['a'] == a) & (df['y'] == y) & (df['z'] == z) & (df['am'] > threshold)])/N
                    p_a_y_z = p_a_and_y_and_z/p_y_z
                    bal += (p_a_y_z - p_a_y)**2
        return bal
    else:
        N = len(df['am'] <= threshold)
        bal = 0
        for a in range(0, 2):
            for y in range(0, 2):
                p_y = len(df[(df['y'] == y) & (df['am'] <= threshold)])/N
                p_a_and_y = len(df[(df['a'] == a) & (
                    df['y'] == y) & (df['am'] <= threshold)])/N
                p_a_y = p_a_and_y/p_y
                for z in range(0, 2):
                    p_y_z = len(df[(df['y'] == y) & (df['z'] == z)
                                   & (df['am'] <= threshold)])/N
                    p_a_and_y_and_z = len(
                        df[(df['a'] == a) & (df['y'] == y) & (df['z'] == z) & (df['am'] <= threshold)])/N
                    p_a_y_z = p_a_and_y_and_z/p_y_z
                    bal += (p_a_y_z - p_a_y)**2
        return bal


def calculate_balance_ratios(df, repaid):
    """Calculates the ratios between the actions dependent on the amounts. Will
    check the ratio of loan  among the top 10 % of the amounts.

    Args:
        df: dataframe containing information about a, y, z and amount of loan
            requested
        repaid: whether or not the observation repaid
    """
    pass


def repeated_cv_fairness(X, y, banker, n_repeats=10, n_folds=10):
    """Calculates various fairness metrics with a repeated k-fold cross
    validation.

    Args:
        X: covariates
        y: response variable
        n_repeats: repetitions of k-fold CV
        n_folds: number of folds to use in CV
    Returns:
        A dictionary of the fairness results.
    """
    amount_threshold = np.median(X['amount'])

    fairness_results = {}
    total_var_dists_y1 = np.zeros(n_repeats*n_folds)
    total_var_dists_y0 = np.zeros(n_repeats*n_folds)
    total_fairness_bal = np.zeros(n_repeats*n_folds)
    total_fairness_bal_low = np.zeros(n_repeats*n_folds)
    total_fairness_bal_high = np.zeros(n_repeats*n_folds)
    gender_balance_y0 = np.zeros(shape=(n_repeats*n_folds, 2))
    gender_balance_y1 = np.zeros(shape=(n_repeats*n_folds, 2))

    t = 0

    for i in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True)
        for train_indices, test_indices in kf.split(X):
            X_train = X.iloc[train_indices, :]
            X_test = X.iloc[test_indices, :]
            y_train = y[train_indices]
            y_test = y[test_indices]

            # fit model
            banker.fit(X_train, y_train)

            num_obs = len(X_test)
            a_obs = np.zeros(num_obs)
            am_obs = np.zeros(num_obs)
            y_obs = np.zeros(num_obs)
            z_obs = np.zeros(num_obs)

            a_obs = banker.get_best_action(X_test)

            for new_obs in range(num_obs):
                obs = X_test.iloc[new_obs]

                z_i = _get_gender(obs)
                y_i = y_test.iloc[new_obs]
                am_i = X.iloc[new_obs]['amount']

                y_obs[new_obs] = y_i
                z_obs[new_obs] = z_i
                am_obs[new_obs] = am_i

            fairness_df = pd.DataFrame(
                {'z': list(z_obs), 'a': list(a_obs), 'y': list(y_obs), 'am': list(am_obs)})

            men = fairness_df.loc[fairness_df['z'] == 1]
            women = fairness_df.loc[fairness_df['z'] == 0]

            z1_y1_a1 = len(men[(men['y'] == 1) & (
                men['a'] == 1)])/len(men[men['y'] == 1])
            z1_y0_a1 = len(men[(men['y'] == 0) & (
                men['a'] == 1)]) / len(men[men['y'] == 0])
            z0_y1_a1 = len(women[(women['y'] == 1) & (
                women['a'] == 1)])/len(women[women['y'] == 1])
            z0_y0_a1 = len(women[(women['y'] == 0) & (
                women['a'] == 1)])/len(women[women['y'] == 0])

            prob_m_y1 = np.array([z1_y1_a1, 1-z1_y1_a1])
            prob_w_y1 = np.array([z0_y1_a1, 1-z0_y1_a1])

            prob_m_y0 = np.array([z1_y0_a1, 1-z1_y0_a1])
            prob_w_y0 = np.array([z0_y0_a1, 1-z0_y0_a1])

            total_var_dists_y1[t] = total_variation(prob_m_y1, prob_w_y1)
            total_var_dists_y0[t] = total_variation(prob_m_y0, prob_w_y0)

            total_fairness_bal[t] = _calculate_balance(fairness_df)
            total_fairness_bal_low[t] = _calculate_balance(
                fairness_df, threshold=amount_threshold, upper=False)
            total_fairness_bal_high[t] = _calculate_balance(
                fairness_df, threshold=amount_threshold, upper=True)

            gender_balance_y0[t] = calculate_balance_ratios(fairness_df, False)
            gender_balance_y1[t] = calculate_balance_ratios(fairness_df, True)
            t = t + 1

    fairness_results['tv0'] = total_var_dists_y0
    fairness_results['tv1'] = total_var_dists_y1
    fairness_results['fair_balance'] = total_fairness_bal
    fairness_results['fair_low'] = total_fairness_bal_low
    fairness_results['fair_high'] = total_fairness_bal_high

    return fairness_results


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


def _get_gender(obs):
    """Gets gender from observation, 1 = male and 0 = female.
    Args:
        obs: covariates from a single observation
    Returns:
        1 if male, 0 if female.
    """
    if obs['marital status_A92'] == 1:
        return 0
    else:
        return 1


def _get_priors(model):
    """Genereates a normally distributed prior for each of the regression
    coefficients with mean = the estimated regression coefficients.

    Args:
        model: the logistic regression model
    Returns:
        The priors
    """
    priors = tfp.distributions.Normal(
        loc=[[i for i in model.coef_[0]]], scale=1)

    return priors


def _get_likelihood(model, X, y_values):
    """Gets the log-likelihood for the data given the model.

    Args:
        model: the probability model used
        X: the covariates
        y_values: the response values
    Returns:
        The log-likelihood
    """
    log_probs = model.predict_log_proba(X)[:, 0]
    log_lik = 0

    for i in range(len(y_values)):
        log_lik += y_values[i]*log_probs[i] + (1-y_values[i])*(1-log_probs[i])

    return log_lik


def fairness(response, interest_rate=0.05):
    """Calculates proportion of a=1 conditional on gender (z) and response (y).

    Args:
        response: name of response variable in the data set
        interest_rate: the interest rate to use
    """
    data = get_data()
    y = data.pop(response)
    X = data

    g_banker = group1_banker.Group1Banker()
    g_banker.set_interest_rate(interest_rate)

    fairness_results = repeated_cv_fairness(
        X, y, g_banker, n_repeats=10, n_folds=5)

    print(f"TVD y=1 = {np.mean(fairness_results['tv1'])}")
    print(f"TVD y=0 = {np.mean(fairness_results['tv0'])}")

    print(f"F_balance = {np.mean(fairness_results['fair_balance'])}")
    print(f"F_balance low = {np.mean(fairness_results['fair_low'])}")
    print(f"F_balance high = {np.mean(fairness_results['fair_high'])}")


def total_variation(prob1, prob2):
    """Calculates the total variation distance by using the formula from
    Wikipedia as referenced in the exercise text.

    Args:
        prob1: probability for a=1 for male (z=1)
        prob2: probability for a=1 for female (z=0)

    """
    return (1/2)*np.sum(np.abs(prob1 - prob2))


if __name__ == "__main__":
    import time
    t0 = time.time()
    np.random.seed(1)
    response = 'repaid'
    fairness(response)
    """
    results = compare_decision_makers(
        n_repeats=20, n_folds=5, response=response, interest_rate=0.05)
    for key in results:
        results[key] = results[key].flatten()
    results = pd.DataFrame(results)

    print(results.describe())
    sns.distplot(results_normal["normal_data_utility"], label="Original data")
    sns.distplot(results_private["private_data_utility"],
                 label="Differentially private data")
    plt.legend()
    plt.xlabel("Average utility over different random train/test draws")
    plt.ylabel("Density")
    plt.show()
    """
