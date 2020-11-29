import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class MedicalData:
    def __init__(self):
        self._get_data()

    def _get_data(self):
        """Gets the data and places it into three dataframes stored as instance
        variables of the object.
        """
        x = pd.read_csv("./data/medical/historical_X.dat",
                        sep=" ", header=None)
        personal_columns = ["sex", "smoker"]
        gene_columns = ["gene " + str(i) for i in range(1, 127)]
        symptom_columns = ["symptom 1", "symptom 2"]
        x_columns = personal_columns + gene_columns + symptom_columns
        x.columns = x_columns

        y = pd.read_csv("./data/medical/historical_Y.dat",
                        sep=" ", header=None)
        y.columns = ["outcome"]

        a = pd.read_csv("./data/medical/historical_A.dat",
                        sep=" ", header=None)
        a.columns = ["action"]

        self.x_train, self.x_test, self.y_train, self.y_test, self.a_train, self.a_test = train_test_split(
            x, y, a, test_size=0.3, random_state=1)

    def data_analysis(self):
        """

        """
        self.frequency_symptoms()
        self.variable_selection(3)

    def _plot_variable_selection(self, accuracy_score1, accuracy_score2, show=False):
        """Plots the accuracy score for the different symptoms.

        Args:
            accuracy_score1: the accuracy score from the CV of symptom 1
            accuracy_score2: the accuracy score from the CV of symptom 2
            show: whether or not to show the plot
        """
        fig, (axis1, axis2) = plt.subplots(1, 2)
        fig.suptitle("Accuracy score for variable selection")

        axis1.plot(range(1, len(accuracy_score1) + 1), accuracy_score1)
        axis1.set_title("Symptom 1")
        axis1.set_ylabel("Accuracy score")
        axis1.set_xlabel("Number of covariates")
        axis2.plot(range(1, len(accuracy_score2) + 1), accuracy_score2)
        axis2.set_title("Symptom 2")
        axis2.set_xlabel("Number of covariates")

        if show:
            plt.show()
        else:
            plt.savefig("img/var_sel.png")

    def variable_selection(self, num_folds):
        """Performs variable selection using a num_folds cross-validation.

        RFECV adapted from https://scikit-learn.org/stable/auto_examples/
        feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-
        examples-feature-selection-plot-rfe-with-cross-validation-py

        Args:
            num_folds: the number of folds to use in the cross-validation
        """
        logistic_regression = LogisticRegression(max_iter=1000)
        variable_selection_cv = RFECV(
            estimator=logistic_regression, step=1, cv=StratifiedKFold(num_folds, random_state=1), scoring='accuracy')

        x = self.x_train.iloc[:, : -2]
        symptom1 = self.x_train.iloc[:, -2]
        symptom2 = self.x_train.iloc[:, -1]

        # symptom 1
        variable_selection_cv.fit(x, symptom1)
        accuracy_symptom1 = variable_selection_cv.grid_scores_
        symptom1_indices = np.where(
            variable_selection_cv.support_ == True)[0]
        print(f"Symptom 1 covariates = {x.columns[symptom1_indices]}")

        # symptom 2
        variable_selection_cv.fit(x, symptom2)
        accuracy_symptom2 = variable_selection_cv.grid_scores_
        symptom2_indices = np.where(
            variable_selection_cv.support_ == True)[0]
        print(f"Symptom 2 covariates = {x.columns[symptom2_indices]}")

        self._plot_variable_selection(accuracy_symptom1, accuracy_symptom2)

    def frequency_symptoms(self, show=False):
        """Generates simple histogram showing the frequency of the different 
        symptoms. Looks at the entire dataset when considering the frequency.

        Args:
            show: whether or not to show the plot
        """
        x_joined = [self.x_train, self.x_test]
        x = pd.concat(x_joined)

        fig, (axis1, axis2) = plt.subplots(1, 2)
        fig.suptitle("Histogram of symptoms")
        axis1.hist(x["symptom 1"], color='b')
        axis1.set_title("Symptom 1")
        axis2.hist(x["symptom 2"], color='r')
        axis2.set_title("Symptom 2")

        if show:
            plt.show()
        else:
            plt.savefig("img/freq_hist.png")

    def measure_effect(self, action):
        """Calculates the measured effect of an action.

        Args:
            action: 1 for treatment and 0 for placebo

        Returns:
            The measured effect.
        """
        y_joined = [self.y_train, self.y_test]
        y = pd.concat(y_joined)
        y_array = self._to_flat_array(y)

        a_joined = [self.a_train, self.a_test]
        a = pd.concat(a_joined)
        a_array = self._to_flat_array(a)

        return self._utility(a_array, y_array, action)

    def measure_effect_symptom(self, action, symptom):
        """Calculates the measured effect of an action.

        Args:
            action: 1 for treatment and 0 for placebo
            symptom: separate observations based on symptom

        Returns:
            The measured effect.
        """
        y_joined = [self.y_train, self.y_test]
        y = pd.concat(y_joined)
        y_array = self._to_flat_array(y)

        a_joined = [self.a_train, self.a_test]
        a = pd.concat(a_joined)
        a_array = self._to_flat_array(a)

        x_joined = [self.x_train, self.x_test]
        x = pd.concat(x_joined)

        if symptom == 1:
            sym_idx = x.iloc[:, -2] == 1
        else:
            sym_idx = x.iloc[:, -1] == 1

        a_cond_sym = a_array[sym_idx]
        y_cond_sym = y_array[sym_idx]

        return self._utility(a_cond_sym, y_cond_sym, action)

    def _to_flat_array(self, df):

        numpy_array = df.to_numpy()
        return numpy_array.flatten()

    def _utility(self, a, y, at):
        """Calculates utility.

        Args:
            a: action array
            y: outcome array
            at: action to measure utility for

        Returns:
            Utility for observation.
        """
        num_at = len(np.where(a == at)[0])
        u = 0

        for i in range(len(a)):
            if a[i] == at and y[i] == 1:
                u += 1

        return u/num_at

    def hierarchical_model(self, data, symptom):
        """Calculates the hierarchical model for the medical data.

        Args:
            data: the data to calculate the posterior probability
            symptom: which symptom to use as response variable

        Returns:
            Posterior probabilites in a Pandas dataframe.
        """
        x = data.iloc[:, : -2]
        if symptom == 1:
            symptom = data.iloc[:, -2]
        else:
            symptom = data.iloc[:, -1]

        num_models = len(x.iloc[0])
        log_likelihoods = np.zeros(num_models + 1)
        model = LogisticRegression(max_iter=500)

        for i in range(0, num_models + 1):
            if i != num_models:
                single_column = x.iloc[:, i].to_numpy()
                single_covariate = single_column.reshape(-1, 1)
                log_reg = model.fit(single_covariate, symptom)
                p_t = log_reg.predict_proba(single_covariate)
                log_likelihoods[i] = -log_loss(symptom, p_t)
            else:
                log_reg = model.fit(x, symptom)
                p_t = log_reg.predict_proba(x)
                log_likelihoods[i] = -log_loss(symptom, p_t)

        # calculating the posterior
        likelihood = np.exp(log_likelihoods)
        prior = np.repeat(1/(num_models + 1), num_models + 1)
        p_y = np.sum(likelihood*prior)
        posterior = (likelihood*prior)/p_y

        # constructing the dataframe
        last_index = pd.Index(data=["all"])
        model_names = x.columns.append(last_index)
        posterior_df = pd.DataFrame(data=posterior, index=model_names)
        posterior_df.columns = ["posterior"]

        return posterior_df

    def _calculate_posterior(self, xtrain, xtest, ytrain, ytest):
        """Calculates the posterior of a test set using a model fitted on 
        training data.

        Args:
            xtrain: training covariates
            xtest: test covariates
            ytrain: training response
            ytest: test response

        Returns:
            The posterior probability of the different models.
        """
        num_models = len(xtrain.iloc[0])
        log_likelihoods = np.zeros(num_models + 1)

        model = LogisticRegression(max_iter=500)

        for i in range(0, num_models + 1):
            if i != num_models:
                single_column = xtrain.iloc[:, i].to_numpy()
                single_covariate = single_column.reshape(-1, 1)
                log_reg = model.fit(single_covariate, ytrain)

                single_column_test = xtest.iloc[:, i].to_numpy()
                single_covariate_test = single_column_test.reshape(-1, 1)
                p_t = log_reg.predict_proba(single_covariate_test)

                log_likelihoods[i] = -log_loss(ytest, p_t)
            else:
                log_reg = model.fit(xtrain, ytrain)

                p_t = log_reg.predict_proba(xtest)
                log_likelihoods[i] = -log_loss(ytest, p_t)

        likelihood = np.exp(log_likelihoods)
        prior = np.repeat(1/(num_models + 1), num_models + 1)
        p_y = np.sum(likelihood*prior)
        posterior = (likelihood*prior)/p_y

        return posterior

    def hierarchical_model_cv(self, symptom, k):
        """Calculates the hierarchical model for the medical data.

        Args:
            symptom: which symptom to use as response variable
            k: the number of folds to use in the cross-validation

        Returns:
            Posterior probabilites in a Pandas dataframe.
        """

        num_models = len(self.x_train.iloc[0]) - 1
        # (folds, models)
        cv_posterior = np.zeros((k, num_models))

        x_joined = [self.x_train, self.x_test]
        x_raw = pd.concat(x_joined)
        x = x_raw.iloc[:, : -2]

        if symptom == 1:
            y = x.iloc[:, -2]
        else:
            y = x.iloc[:, -1]

        kf = KFold(n_splits=k, shuffle=True)
        k_counter = 0

        for train_indices, test_indices in kf.split(x):
            xtrain = x.iloc[train_indices, :]
            ytrain = y[train_indices]
            xtest = x.iloc[test_indices, :]
            ytest = y[test_indices]

            posterior = self._calculate_posterior(xtrain, xtest, ytrain, ytest)
            cv_posterior[k_counter, :] = posterior
            k_counter += 1

        posterior = np.mean(cv_posterior, 0)

        # constructing the dataframe
        last_index = pd.Index(data=["all"])
        model_names = x.columns.append(last_index)
        posterior_df = pd.DataFrame(data=posterior, index=model_names)
        posterior_df.columns = ["posterior"]

        return posterior_df


def plot_posteriors(posteriors, num, title, show=True):
    """Plots the top k posterios.

    Args:
        posteriors: sorted Pandas dataframe with posteriors
        num: the number of posteriors to plot
        title: title of the histogram
        show: whether or not to show the plot
    """
    plot_posteriors = posteriors.sort_values(
        by="posterior", ascending=False)[:num]

    plot_posteriors.plot.bar()
    plt.title(title)
    plt.xlabel("covariates")
    plt.ylabel("P(model | y)")

    if show:
        plt.show()
    else:
        filename = title.replace(" ", "_") + ".png"
        plt.savefig("img/" + filename)


if __name__ == "__main__":
    data = MedicalData()
    # data.data_analysis()
    # expected_utility_1 = data.measure_effect(1)
    # expected_utility_0 = data.measure_effect(0)
    # print(f"E[U|a_t = 1] = {expected_utility_1}")
    # print(f"E[U|a_t = 0] = {expected_utility_0}")

    # util_sym1_a1 = data.measure_effect_symptom(1, 1)
    # util_sym2_a1 = data.measure_effect_symptom(1, 2)
    # util_sym1_a0 = data.measure_effect_symptom(0, 1)
    # util_sym2_a0 = data.measure_effect_symptom(0, 2)
    # print(f"E[U|a_t = 1, sym = 1] = {util_sym1_a1}")
    # print(f"E[U|a_t = 1, sym = 2] = {util_sym2_a1}")
    # print(f"E[U|a_t = 0, sym = 1] = {util_sym1_a0}")
    # print(f"E[U|a_t = 0, sym = 2] = {util_sym2_a0}")

    x_joined = [data.x_train, data.x_test]
    x = pd.concat(x_joined)
    sym1_posteriors = data.hierarchical_model(x, 1)
    # plot_posteriors(sym1_posteriors, 5, "histogram for symptom 1", show=False)

    # sym2_posteriors = data.hierarchical_model(x, 2)
    # plot_posteriors(sym2_posteriors, 5, "histogram for symptom 2", show=False)

    # sym1_cv_posteriors = data.hierarchical_model_cv(1, 5)
    # plot_posteriors(sym1_cv_posteriors, 5,
    #                 "histogram cv symptom 1", show=False)

    # sym2_cv_posteriors = data.hierarchical_model_cv(2, 5)
    # plot_posteriors(sym2_cv_posteriors, 5,
    #                 "histogram cv symptom 2", show=False)

    # part 2
    from random_recommender import RandomRecommender
    rr = RandomRecommender(1, 1)
    x_joined = [data.x_train, data.x_test]
    x = pd.concat(x_joined)
    a_joined = [data.a_train, data.a_test]
    a = pd.concat(a_joined)
    y_joined = [data.y_train, data.y_test]
    y = pd.concat(y_joined)
    ur = rr.estimate_utility(x, a, y)
    print(f"Average utility = {round(ur, 4)}")

    from historical_recommender import HistoricalRecommender
    hr = HistoricalRecommender(2, 2)
    uh = hr.estimate_utility(x, a, y)
    print(f"Average utility = {round(uh, 4)}")

    from HistoricalPolicy import HistoricalPolicy
    hp = HistoricalPolicy(2, 2, a, y)
    #hp.method0(100, 1000, 5, 0.05)

    #hr.estimate_utility(x, a, y, hp)
    print(f"pi0_hat = {hp.pi0_hat}")
    print(f"theta_0 = {hp.theta_hat[0]}")
    print(f"theta_1 = {hp.theta_hat[1]}")

    # bootstrap
    #boot_util = hp.bootstrap_expected_utility(500)
    #hp.plot_bootstrap_hist(boot_util, 500)
    hp.bootstrap_percentile(100, 5, 0.05)
    hp.plot_bootstrap_ci(100, 5)
