from sklearn import linear_model
import numpy as np
import attr
from sklearn.linear_model import LogisticRegression
from part1 import MedicalData
from utilities import FinalAnalysis, FixedTreatmentPolicy, RecommenderModel
from martin_historical_recommender import RecommenderModel
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder


@attr.s
class Approach1_impr_bl(RecommenderModel):

    def fit_treatment_outcome(self, data, actions, outcomes):

        self.actions = actions
        self.outcome = outcomes
        self.data = data

        action_matrix = self.get_action_matrix(len(data), actions)

        x = np.hstack((data.copy(), action_matrix))
        regression_model = LogisticRegression(max_iter=5000, n_jobs=-1)

        regression_model.fit(x, outcomes.flatten())
        self.model = regression_model

        policy_model = LogisticRegression(max_iter=5000, n_jobs=-1)
        policy_model.fit(data, actions.flatten())
        self.policy = policy_model

    def get_action_probabilities(self, user_data):
        if isinstance(user_data, pd.core.series.Series):
            user_data = user_data.to_numpy().reshape(1, -1)
        else:
            user_data = user_data.reshape(1, -1)

        # pi(a|x)
        pi = np.zeros(self.n_actions)
        expected_reward = np.zeros(self.n_actions)

        for a_t in range(self.n_actions):
            expected_reward[a_t] = self.estimate_expected_conditional_reward(
                user_data, a_t)

        pi[np.argmax(expected_reward)] = 1

        assert np.sum(pi) == 1

        return pi

    def predict_proba(self, data, treatment):
        """Predicts the probability of y.

        Args:
            data: x_t
            treatment: a_t

        Returns:
            P(y | a, x).
        """
        treatment_vector = self.get_treatment_vector(treatment)

        if isinstance(data, pd.core.series.Series):
            x_t = data.to_numpy()
            user_array = np.hstack((x_t, treatment_vector)).reshape(1, -1)
        else:
            user_array = np.hstack((data.flatten(), treatment_vector)).reshape(1, -1)

        # P(y_t | a_t, x_t)
        p = self.model.predict_proba(user_array)
        return p[0]

    def estimate_expected_conditional_reward(self, user_data, action):
        """Estimates the expected reward conditional on an action.

        Args:
            user_data: the covariates of an observation
            action: the selected action (a_t = a)

        Returns:
            The expected conditional reward E[r_t | a_t = a].
        """
        estimated_cond_utility = 0

        # sum over possible outcomes for expected reward
        for y_t in range(self.n_outcomes):
            p_y = self.predict_proba(user_data.copy(), action)
            estimated_cond_utility += p_y[y_t] * self.reward(action, y_t)

        return estimated_cond_utility

    def observe(self, user, action, outcome):
        """Updates the data.

        Args:
            user: x_t
            action: a_t
            outcome: y_t
        """
        self.data = np.vstack((self.data, user))
        self.actions = np.vstack((self.actions, action))
        self.outcome = np.vstack((self.outcome, outcome))


@attr.s
class Approach1_impr_varsel(RecommenderModel):

    def fit_treatment_outcome(self, data, actions, outcomes):

        self.actions = actions
        self.outcome = outcomes
        self.data = data

        regression_model = LogisticRegression(max_iter=5000, n_jobs=-1)

        variable_selection_cv = RFECV(
            estimator=regression_model, step=1, cv=StratifiedKFold(5, random_state=1, shuffle=True), n_jobs=-1, scoring='accuracy')

        variable_selection_cv.fit(self.data, self.outcome.flatten())

        # covariates from variable selection
        selected_var = variable_selection_cv.support_
        self.selected_variables = selected_var

        action_matrix = self.get_action_matrix(len(data), actions)

        if isinstance(data, pd.DataFrame):
            x_selected = np.hstack((data.iloc[:, selected_var], action_matrix))
        else:
            x_selected = np.hstack((data[:, selected_var], action_matrix))

        regression_model.fit(
            x_selected, outcomes.flatten())
        self.model = regression_model

    def get_action_probabilities(self, user_data):
        # print("Recommending")
        if isinstance(user_data, pd.core.series.Series):
            user_data = user_data.to_numpy().reshape(1, -1)
        else:
            user_data = user_data.reshape(1, -1)

        # pi(a|x)
        pi = np.zeros(self.n_actions)
        expected_reward = np.zeros(self.n_actions)

        for a_t in range(self.n_actions):
            expected_reward[a_t] = self.estimate_expected_conditional_reward(
                user_data, a_t)

        pi[np.argmax(expected_reward)] = 1

        return pi

    def predict_proba(self, data, treatment):
        """Predicts the probability of y.

        Args:
            data: x_t
            treatment: a_t

        Returns:
            P(y | a, x).
        """
        treatment_vector = self.get_treatment_vector(treatment)

        if isinstance(data, pd.core.series.Series):
            x_t = data.to_numpy()
            user_array = np.hstack((x_t, treatment_vector)).reshape(1, -1)
        else:
            user_array = np.hstack((data.flatten(), treatment_vector)).reshape(1, -1)

        # P(y_t | a_t, x_t)
        selected = np.hstack((self.selected_variables, np.repeat(True, self.n_actions)))
        user_data = user_array.flatten()
    
        p = self.model.predict_proba(user_data[selected].reshape(1, -1))
        return p[0]

    def estimate_expected_conditional_reward(self, user_data, action):
        """Estimates the expected reward conditional on an action.

        Args:
            user_data: the covariates of an observation
            action: the selected action (a_t = a)

        Returns:
            The expected conditional reward E[r_t | a_t = a].
        """
        estimated_cond_utility = 0

        # sum over possible outcomes for expected reward
        for y_t in range(self.n_outcomes):
            p_y = self.predict_proba(user_data.copy(), action)
            estimated_cond_utility += p_y[y_t] * self.reward(action, y_t)

        return estimated_cond_utility

    def observe(self, user, action, outcome):
        """Updates the data.

        Args:
            user: x_t
            action: a_t
            outcome: y_t
        """
        self.data = np.vstack((self.data, user))
        self.actions = np.vstack((self.actions, action))
        self.outcome = np.vstack((self.outcome, outcome))


class ImprovedRecommender:

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    # By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        return outcome

    # Set the reward function r(a, y)
    def set_reward(self, reward):
        self.reward = reward

    ##################################
    # Fit a model from patient data.
    #
    # This will generally speaking be an
    # unsupervised model. Anything from a Gaussian mixture model to a
    # neural network is a valid choice.  However, you can give special
    # meaning to different parts of the data, and use a supervised
    # model instead.
    def fit_data(self, data):
        print("Preprocessing data")
        return None

    # Fit a model from patient data, actions and their effects
    # Here we assume that the outcome is a direct function of data and actions
    # This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcome, recommender_model=None):
        """Fits historical data to the policy. Includes the action as a
        covariate in the logistic regression model.

        Args:
            data: covariates, x_t
            actions: the action taken for each observation
            outcome: the outcome y_t | a_t, x_t
        """
        print("Fitting treatment outcomes")

        x = data.copy()
        if isinstance(data, pd.DataFrame):
            x = x.to_numpy()

        if isinstance(actions, pd.DataFrame):
            actions = actions.to_numpy()

        if isinstance(outcome, pd.DataFrame):
            outcome = outcome.to_numpy()

        x = np.hstack((x, actions))

        if recommender_model is None:
            recommender_model = Approach1_impr_bl(
                self.n_actions, self.n_outcomes)

        recommender_model.fit_treatment_outcome(data, actions, outcome)
        recommender_model.set_reward(self.reward)
        self.recommender_model = recommender_model

    # Estimate the utility of a specific policy from historical data (data, actions, outcome),
    # where utility is the expected reward of the policy.
    ##
    # If policy is not given, simply use the average reward of the observed actions and outcomes.
    ##
    # If a policy is given, then you can either use importance
    # sampling, or use the model you have fitted from historical data
    # to get an estimate of the utility.
    ##
    # The policy should be a recommender that implements get_action_probability()
    def estimate_utility(self, data, actions, outcome, policy=None):
        T = len(actions)
        print(f"Estimating = {T} observations")

        utility = np.zeros(T)

        for t in range(T):
            print(f"{t}/{T}")

            # one observation
            user_data = data.iloc[t]

            # action distribution
            pi_a_x = self.get_action_probabilities(user_data)

            # expected reward
            utility[t] = self.estimate_expected_reward(user_data, pi_a_x)

        return np.mean(utility)

    def estimate_expected_reward(self, user_data, pi):
        """Estimates the expected reward for a given observation.

        Args:
            user_data: the observation data (x_t)
            pi: the conditional distribution pi(a_t | x_t)

        Returns:
            The expected reward for the observation.
        """
        estimated_utility = 0

        for a_t in range(self.n_actions):

            for y_t in range(self.n_outcomes):
                p_y = self.predict_proba(user_data.copy(), a_t)

                estimated_utility += pi[a_t] * p_y[y_t] * self.reward(a_t, y_t)

        return estimated_utility

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes

    def predict_proba(self, data, treatment):
        """Calculates the conditional probability P(y | a, x).

        """
        return self.recommender_model.predict_proba(data, treatment)

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        """Calculates the conditional distribution of actions pi(a_t | x_t).

         Args:
             user_data: observation to calculate the conditional distribution for
         """
        return self.recommender_model.get_action_probabilities(user_data)

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        return np.random.choice(self.n_actions, p=self.get_action_probabilities(user_data))

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        """Observe new observations dynamically. The improved recommender
        will not utilize this, but saves the data.

        Args:
            user: covariates for a specific user
            action: the action selected by the recommender/policy
            outcome: the outcome y | a
        """
        self.recommender_model.observe(user, action, outcome)

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self, n_tests=1000, generator=None):
        analysis = FinalAnalysis()
        # 1
        print(f"1. Specific fixed treatment policy")
        action_utilities = analysis.fixed_treatment_policy_check(
            recommender=self, n_tests=n_tests, generator=generator)
        for a_t in range(self.n_actions):
            print(f"E[U | a = {a_t}] = {round(action_utilities[a_t, 1], 4)}")

        # 2
        print(f"2. Looking at specific genes more closely")

        return None


if __name__ == "__main__":
    data = MedicalData()
    n_actions = len(np.unique(data.a_train))
    n_outcomes = len(np.unique(data.y_train))

    # im_recommender = ImprovedRecommender(n_actions, n_outcomes)
    # im_recommender.set_reward(lambda a, y: y - 0.1*(a != 0))
    # im_recommender.fit_treatment_outcome(
    #     data.x_train, data.a_train, data.y_train)

    # im_estimated_utility = im_recommender.estimate_utility(
    #     data.x_test, data.a_test, data.y_test)
    # print(f"Estimated expected utility = {round(im_estimated_utility, 4)}")

    im_recommender_varsel = ImprovedRecommender(n_actions, n_outcomes)
    im_recommender_varsel.set_reward(lambda a, y: y - 0.1*(a != 0))
    varsel_model = Approach1_impr_varsel(
        im_recommender_varsel.n_actions, im_recommender_varsel.n_outcomes)
    im_recommender_varsel.fit_treatment_outcome(
        data.x_train, data.a_train, data.y_train, varsel_model)

    im_varsel_estimated_utility = im_recommender_varsel.estimate_utility(
        data.x_test, data.a_test, data.y_test)
    print(
        f"Estimated expected utility = {round(im_varsel_estimated_utility, 4)}")
