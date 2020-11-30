from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import numpy as np
from part1 import MedicalData
import pandas as pd


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
        self.internal_policy = Approach1_policy(
            self.n_actions, self.n_outcomes)

    # Fit a model from patient data, actions and their effects
    # Here we assume that the outcome is a direct function of data and actions
    # This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcome):
        print("Fitting treatment outcomes")
        return None

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

        if policy is not None:
            if policy.var_sel:
                selected_data = data[data.columns[policy.selected_variables]]
            else:
                selected_data = data

            for t in range(T):
                # one observation
                user_data = selected_data.iloc[t]

                # get estimated best action
                a_t = np.random.choice(
                    self.n_actions, p=policy.get_action_probabilities(user_data))

                # calculate utility from estimated action and outcome
                #utility[t] = self.reward(a_t, outcome.iloc[t])
                utility[t] = policy.expected_reward(
                    a_t, policy.predict_proba(user_data, a_t))

            return np.mean(utility)
        else:
            T = len(actions)
            a = actions.to_numpy()
            y = outcome.to_numpy()
            utility = np.empty(T)

            for t in range(T):
                if a[t] == 0:
                    utility[t] = y[t]
                else:
                    utility[t] = -0.1*a[t] + y[t]

            return np.mean(utility)

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        """Calculates P(y|a = treatment, x = data) and returns the distribution
        of effects/outcomes (y).

        Args:
            data: the covariates (x_t) for an observation
            treatment: the action (a_t) used to predict the outcome

        Returns:
            The probabilities for the outcomes.
        """
        return self.internal_policy.predict_proba(data, treatment)

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        return self.internal_policy.get_action_probabilities(user_data)

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        return np.random.choice(self.n_actions, p=self.get_action_probabilities(user_data))

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        return None

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None


class Approach1_policy:

    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes

    def fit_data(self, data, var_sel=False):
        self.var_sel = var_sel
        print("Preprocessing data")
        y = data.pop('y')
        a = data.pop('a')
        x = data

        regression_model = LogisticRegression(max_iter=1000, n_jobs=-1)

        if var_sel:
            print("Variable selection")
            variable_selection_cv = RFECV(
                estimator=regression_model, step=1, cv=StratifiedKFold(5, random_state=1, shuffle=True), n_jobs=-1, scoring='accuracy')
            variable_selection_cv.fit(x, y)

            # covariates from variable selection
            selected_var = variable_selection_cv.support_
            self.selected_variables = selected_var

            print(
                f"- number of variables: {variable_selection_cv.n_features_}")

            # use only the variables selected
            x_selected = pd.DataFrame(x[x.columns[selected_var]])
        else:
            x_selected = x

        x_selected['a'] = a
        self.model = regression_model.fit(x_selected, y)

    def expected_reward(self, action, y_prob):
        """Estimates expected reward.

        Args:
            action: the action to condition on
            y_prob: the probability for the different outcomes (0/1)

        Returns:
            The expected reward.
        """
        return -0.1*action + (0*y_prob[0] + 1*y_prob[1])

    def predict_proba(self, data, treatment):
        """Calculates P(y|a = treatment, x = data) and returns the distribution
        of effects/outcomes (y).

        Args:
            data: the covariates (x_t) for an observation
            treatment: the action (a_t) used to predict the outcome

        Returns:
            The probabilities for the outcomes.
        """
        # add column with action to the observation
        data["a"] = treatment
        user_data = data.to_numpy().reshape(1, -1)
        y_prob = self.model.predict_proba(user_data)

        return y_prob[0]

    def get_action_probabilities(self, user_data):
        e_r = np.zeros(self.n_actions)

        for a_t in range(self.n_actions):
            y_prob = self.predict_proba(user_data, a_t)
            e_r[a_t] = self.expected_reward(a_t, y_prob)

        action_prob = np.zeros(self.n_actions)

        # the recommended action
        action_prob[e_r.argmax()] = 1
        return action_prob/np.sum(action_prob)
