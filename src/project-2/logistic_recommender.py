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


class LogisticRecommender:

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

    def fit_treatment_outcome(self, data, actions, outcome):
        """ Group the data by action and fit a logistic regression model for each possible action.

        The models are saved in the class attribute "models" as a numpy.array
        of models, indexed by the action value.

        Args:
            data: numpy.ndarray with observed features
            actions: numpy.array with preformed actions
            outcome: numpy.array with observed outcomes
        """
        self.models = np.empty(self.n_actions)

        for a in range(self.n_actions):
            ind = actions == a

            self.models[a] = LogisticRegression(
                penalty='l2', multi_class='ovr')
            self.models[a].fit(data[ind], outcome[ind])

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
        return 0

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        return self.models[treatment].predict_proba(data)[:, 1]

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

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        pass

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None
