from sklearn import linear_model
import numpy as np
from sklearn.linear_model import LogisticRegression
from part1 import MedicalData
from utilities import RecommenderModel, FinalAnalysis, FixedTreatmentPolicy
import pandas as pd
import attr


@attr.s
class Approach1_hist_bl(RecommenderModel):

    def fit_treatment_outcome(self, data, actions, outcomes):
        """Fits the model used from historical data.

        Args:
            data: x_t
            actions: a_t | x_t
            outcomes: y_t | a_t, x_t
        """

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
        """Calculates the action probabilities using a stochastic policy.

        Args:
            user_data: an observation x_t
        
        Returns
            The distribution pi(a|x).
        """
        if isinstance(user_data, pd.core.series.Series):
            user_data = user_data.to_numpy().reshape(1, -1)
        else:
            user_data = user_data.reshape(1, -1)

        # pi(a|x)
        pi = np.zeros(self.n_actions)

        # predict values for a
        predictions = self.policy.predict_proba(user_data)

        for a_t in range(len(predictions[0])):
            pi[a_t] = predictions[0][a_t]

        return pi

    def predict_proba(self, data, treatment):
        """Estimates P(y|a, x).

        Args:
            data: x_t
            treatment: a_t
        
        Returns
            An array of probabilities indexed by y_t.
        """
        treatment_vector = self.get_treatment_vector(treatment)

        if isinstance(data, pd.core.series.Series):
            x_t = data.to_numpy()
            user_array = np.hstack((x_t, treatment_vector)).reshape(1, -1)
        else:
            user_array = np.hstack((data.flatten(), treatment_vector)).reshape(1, -1)

        p = self.model.predict_proba(user_array)

        return p[0]

    def observe(self, user, action, outcome):
        """Stores the observations, but does not update not the model.

        Args:
            user: x_t
            action: a_t
            outcome: y_t
        """
        self.data = np.vstack((self.data, user))
        self.actions = np.vstack((self.actions, action))
        self.outcome = np.vstack((self.outcome, outcome))


class HistoricalRecommender:

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes):
        """Constructor for the historical recommender class.

        Args:
            n_actions: the number of actions the recommender can choose from (a)
            n_outcomes: the number of possible outcomes (y)
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self.all_p = list()

    # By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        """Sets the default reward equal to the outcome.

        Args:
            action: a
            outcome: y
        
        Returns
            y.
        """
        return outcome

    # Set the reward function r(a, y)
    def set_reward(self, reward):
        """Sets a specific reward function.

        Args:
            reward: a function accepting action and outcome (a, y) in order to
                calculate the reward
        """
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
        """Fits an unsupervised model to the data.

        Args:
            data: the observations (x_t)
        """
        print("Preprocessing data")
        return None

    # Fit a model from patient data, actions and their effects
    # Here we assume that the outcome is a direct function of data and actions
    # This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcome, recommender_model=None):
        """Calculates and fits:

        P(y_t | a_t, x_t): the distribution of outcomes
        pi(a_t | x_t): the distribution of actions

        Args:
            data: the covariates of the historical data
            actions: the vector of actions a_t
            outcome: the vector of outcomes y_t
        """
        print("Fitting treatment outcomes")

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

        if recommender_model is None:
            recommender_model = Approach1_hist_bl(
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
        """Estimates the expected utility for the provided data set.

        Args:
            data: covariates of observations
            actions: vector of action taken for the observations
            outcome: vector of outcomes for the observations

        Returns:
            The estimated expected utility.
        """
        # if policy is not given, use the class' own get_action_probabilities
        if policy is None:
            action_prob_method = self.get_action_probabilities
        # else, use the provided policy
        else:
            action_prob_method = policy.get_action_probabilities

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        T = len(actions)
        print(f"Estimating = {T} observations")

        utility = np.zeros(T)

        for t in range(T):
            # one observation
            user_data = data[t, ]

            # action distribution
            pi_a_x = action_prob_method(user_data)

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
        """Calculates the conditional probability P(y | a, x) through the 
        recommender model.

        Args:
            data: the covariates for an observation (x_t)
            treatment: the action to condition on (a_t)

        Returns:
            The conditional distribution.
        """
        if isinstance(data, pd.core.series.Series):
            data = data.to_numpy()

        return self.recommender_model.predict_proba(data, treatment)

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        """Calculates the conditional distribution of actions pi(a_t | x_t).

        Args:
            user_data: observation to calculate the conditional distribution for  

        Returns
            The probabilities for different actions a.           
        """
        pi = self.recommender_model.get_action_probabilities(user_data)
        self.all_p.append(pi)
        return pi

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)

    def recommend(self, user_data):
        """Recommends an action based on x_t.

        Args:
            user_data: x_t
        
        Returns
            An action a_t.
        """
        return np.random.choice(self.n_actions, p=self.get_action_probabilities(user_data))

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        """Observe new observations dynamically. The historical recommender
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
        """After the testing and the new observations are added.

        Args:
            n_tests: number of tests to run
            generator: the generator from Christos' test bench
        """
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
    np.random.seed(1)
    data = MedicalData()
    n_actions = len(np.unique(data.a_train))
    n_outcomes = len(np.unique(data.y_train))

    hist_recommender = HistoricalRecommender(n_actions, n_outcomes)
    hist_recommender.set_reward(lambda a, y: y - 0.1*(a != 0))
    hist_recommender.fit_treatment_outcome(
        data.x_train, data.a_train, data.y_train)

    hist_estimated_utility = hist_recommender.estimate_utility(
        data.x_test, data.a_test, data.y_test)
    print(f"Estimated expected utility = {round(hist_estimated_utility, 4)}")
    p_star = np.min(hist_recommender.all_p)
    print(f"p_star = {round(p_star, 4)}")
    print(f"n = {len(data.x_test)}")
