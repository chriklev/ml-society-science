import TestImplementation
import group1_banker
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold


def total_variation(prob1, prob2):
    """Calculates the total variation distance by using the formula from
    Wikipedia as referenced in the exercise text.

    Args:
        prob1: probability for a=1 for male (z=1)
        prob2: probability for a=1 for female (z=0)

    """
    return (1/2)*np.sum(np.abs(prob1 - prob2))


def fairness(response, interest_rate=0.05):
    """Calculates proportion of a=1 conditional on gender (z) and response (y).

    Args:
        response: name of response variable in the data set
        interest_rate: the interest rate to use
    """
    data = TestImplementation.get_data()
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
            t = t + 1

    fairness_results['tv0'] = total_var_dists_y0
    fairness_results['tv1'] = total_var_dists_y1
    fairness_results['fair_balance'] = total_fairness_bal
    fairness_results['fair_low'] = total_fairness_bal_low
    fairness_results['fair_high'] = total_fairness_bal_high

    return fairness_results


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


def countplot():
    # Get data
    X = TestImplementation.get_data()
    y = X.pop("repaid")

    # Fit the banker
    banker = group1_banker.Group1Banker()
    banker.set_interest_rate(.05)
    banker.fit(X, y)

    # Get predictions
    y_predicted = banker.get_best_action(X)
    print(y_predicted.shape)

    is_female = X["marital status_A92"] == 1
    sex = pd.Series(is_female.map({True: "female", False: "male"}))

    gender_data = pd.DataFrame()
    gender_data["repaid"] = pd.concat((y, pd.Series(y_predicted))).map(
        {0: "no", 1: "yes"})
    gender_data["response"] = np.repeat(["true", "predicted"], y.size)
    gender_data["sex"] = pd.concat((sex, sex))

    sns.set_style(style="whitegrid")
    g = sns.catplot(x="repaid", hue="sex", col="response",
                    data=gender_data, kind="count",
                    height=4, aspect=.7)
    plt.savefig("img/gender_compare.png")
    plt.show()


def check_gender_significance():
    import statsmodels.api as sm

    X = TestImplementation.get_data()
    y = X.pop("repaid")

    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary2())


if __name__ == "__main__":
    # countplot()
    check_gender_significance()
    np.random.seed(1)
    response = 'repaid'
    fairness(response)
