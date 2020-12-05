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

from sklearn.linear_model import LogisticRegression
import numpy as np


class AdaptiveLogisticRecommender:

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
        self.models = None

    # By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        return outcome

    def set_reward(self, reward):
        """ Set the reward function r(a, y)

        Args:
            reward: a function calcultaing reward and taking the arguments (action, outcome)
        """
        self.reward = reward

    def fit_treatment_outcome(self, data, actions, outcome):
        """ Group the data by action and fit a logistic regression model for each possible action.

        The models are saved in the class attribute "models" as a numpy.array
        of models, indexed by the action value.

        Args:
            data: numpy.ndarray with observed features
            actions: numpy.array with preformed actions
            outcome: numpy.array with observed outcomes
        """
        # Ensure actions and outcomes are in 1-d vectors
        actions = actions.reshape(-1)
        outcome = outcome.reshape(-1)
        # Save training data
        self.model_data = data
        self.model_actions = actions
        self.model_outcome = outcome

        self.models = {}

        for a in range(self.n_actions):
            ind = actions == a

            self.models[a] = LogisticRegression(
                penalty='l2', max_iter=200)
            action_data = data[ind, :]
            action_outcome = outcome[ind]
            # See if there are any observations with current action.
            if action_data.shape[0] > 1 and np.unique(action_outcome).size > 1:
                self.models[a].fit(action_data, outcome[ind])
            # If not, create placeholder fit to not break the other methods.
            else:
                self.models[a].fit(
                    np.zeros((2, action_data.shape[1])), np.array([0, 1]))

    def estimate_utility(self, data, actions, outcome):
        """ Calculates estimated utility for this reccomender.

        Args:
            data: numpy.ndarray with observed features
            actions: numpy.array with preformed actions
            outcome: numpy.array with observed outcomes

        Returns:
            Estimated utility as a float.
        """
        self.fit_treatment_outcome(data, actions, outcome)

        utility_total = 0
        # For each observation
        for i in range(len(outcome)):
            # Find expected reward for each action
            action_rewards = np.empty(self.n_actions)
            for a in range(self.n_actions):
                action_rewards[a] = self.reward(
                    a, self.predict_proba(data[i, :], a))

            utility_total += np.max(action_rewards)

        return utility_total/len(actions)

    def predict_proba(self, data, treatment):
        """ Predicts the distribution of outcomes given features and a treatment.

        Args:
            data: numpy.array with observed features.
            treatment: integer representing the treatment.
        """
        return self.models[treatment].predict_proba(data.reshape((-1, 130)))[0, 1]

    def get_action_probabilities(self, user_data):
        """ Finds action probabilities by expected reward.

        Args:
            user_data: numpy.array with the features for a single observation.

        Returns:
            np.array with distribution of recommendations.
        """
        utils = np.empty(self.n_actions)
        for a in range(self.n_actions):
            utils[a] = self.reward(a, self.predict_proba(user_data, a))

        # Squared values to emphasize difference in rewards
        return utils**2/(utils**2).sum()

    def recommend(self, user_data):
        """ Gives recommendation for a specific user.

        Args:
            user_data: numpy.array with the features for a single observation.

        Returns:
            Recommended action as integer.
        """
        return np.argmax(self.get_action_probabilities(user_data))

    def observe(self, user, action, outcome):
        """ Update logistic model with new observation.

        Args:
            data: numpy.array with observed features
            actions: preformed action as integer
            outcome: observed outcome as integer
        """
        if self.models is None:
            new_data = user.reshape((1, -1))
            new_actions = np.array(action).reshape((1))
            new_outcome = np.array(outcome).reshape((1))
        else:
            user = user.reshape((1, -1))
            new_data = np.append(self.model_data, user, axis=0)
            new_actions = np.append(self.model_actions, action)
            new_outcome = np.append(self.model_outcome, outcome)

        self.fit_treatment_outcome(new_data, new_actions, new_outcome)

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None
