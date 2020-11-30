# -*- Mode: python -*-
# A simple reference recommender
#
#
# This is a medical scenario with historical data.
#
# General functions
#
# - set_reward
#
# There is a set of functions for dealing with historical data:
#
# - fit_data
# - fit_treatment_outcome
# - estimate_utiltiy
#
# There is a set of functions for online decision making
#
# - predict_proba
# - recommend
# - observe

from sklearn import linear_model
import numpy as np
from sklearn.linear_model import LogisticRegression
from part1 import MedicalData
import pandas as pd


class AdaptiveRecommender:

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
    def fit_treatment_outcome(self, data, actions, outcome):
        """Fits historical data to the policy. Includes the action as a 
        covariate in the logistic regression model.

        Args:
            data: covariates, x_t
            actions: the action taken for each observation
            outcome: the outcome y_t | a_t, x_t
        """
        print("Fitting treatment outcomes")

        if self.n_actions != len(np.unique(actions)):
            print(
                f"Warning: self.n_actions = {self.n_actions}, len(np.unique(actions)) = {len(np.unique(actions))}")
            print(f"self.n_actions will be set to the actual number of actions")
            self.n_actions = np.unique(actions)

        regression_model = LogisticRegression(max_iter=5000, n_jobs=-1)

        x = data.copy()
        if isinstance(data, pd.DataFrame):
            x = x.to_numpy()

        if isinstance(actions, pd.DataFrame):
            actions = actions.to_numpy()

        if isinstance(outcome, pd.DataFrame):
            outcome = outcome.to_numpy()

        self.data = x
        self.actions = actions
        self.outcome = outcome

        x = np.hstack((x, actions))

        regression_model.fit(x, outcome.flatten())
        self.model = regression_model

        policy_model = LogisticRegression(max_iter=5000, n_jobs=-1)
        policy_model.fit(data, actions.flatten())
        self.policy = policy_model

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
            # one observation
            user_data = data.iloc[t]

            # action distribution
            pi_a_x = self.get_action_probabilities(user_data)

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
                p_y = self.predict_proba(user_data, a_t)

                estimated_utility += pi[a_t] * p_y[y_t] * self.reward(a_t, y_t)

        return estimated_utility

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes

    def predict_proba(self, data, treatment):
        """Calculates the conditional probability P(y | a, x).

        """
        data['a'] = treatment
        user_array = data.to_numpy().reshape(1, -1)

        p = self.model.predict_proba(user_array)
        return p[0]

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        # print("Recommending")
        if isinstance(user_data, pd.core.series.Series):
            user_data = user_data.to_numpy().reshape(1, -1)
        else:
            user_data = user_data.reshape(1, -1)

        pi = self.policy.predict_proba(user_data)
        return pi[0]

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        # breakpoint()
        return np.random.choice(self.n_actions, p=self.get_action_probabilities(user_data))

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        """Updates the model based on new observation.

        """
        self.data = np.vstack((self.data, user))
        self.actions = np.vstack((self.actions, action))
        self.outcome = np.vstack((self.outcome, outcome))

        x = np.hstack((self.data, self.actions))
        self.model = self.model.fit(x, self.outcome.flatten())

        self.policy = self.policy.fit(self.data, self.actions.flatten())

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None


if __name__ == "__main__":
    data = MedicalData()
    n_actions = len(np.unique(data.a_train))
    n_outcomes = len(np.unique(data.y_train))

    ada_recommender = AdaptiveRecommender(n_actions, n_outcomes)
    ada_recommender.set_reward(lambda a, y: y - 0.1*(a != 0))
    ada_recommender.fit_treatment_outcome(
        data.x_train, data.a_train, data.y_train)

    ada_estimated_utility = ada_recommender.estimate_utility(
        data.x_test, data.a_test, data.y_test)
    print(f"Estimated expected utility = {round(ada_estimated_utility, 4)}")
