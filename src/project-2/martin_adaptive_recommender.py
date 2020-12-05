from sklearn import linear_model
import numpy as np
from sklearn.linear_model import LogisticRegression
from part1 import MedicalData
import pandas as pd
import attr
from martin_historical_recommender import RecommenderModel
from scipy.stats import norm, uniform
from scipy.optimize import minimize
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


@attr.s
class Approach1_adap_bl(RecommenderModel):

    def fit_treatment_outcome(self, data, actions, outcomes):
        """Fits the model to the historical data.

        Args:
            data: covariates x_t for each observation
            actions: the action taken for historical observations a_t
            outcomes: the observed outcomes for historical observations y_t
        """
        self.outcome = outcomes
        self.data = data

        action_matrix = self.get_action_matrix(len(data), actions)
        self.actions = action_matrix

        x = np.hstack((data.copy(), action_matrix))
        regression_model = LogisticRegression(max_iter=5000, n_jobs=-1)

        regression_model.fit(x, outcomes.flatten())
        self.model = regression_model

    def get_action_probabilities(self, user_data):
        """Returns the action probabilities, will be 1 for the action with the 
        highest expected reward.

        Args:
            user_data: observation x_t

        Returns
            An array of probabilities for the different actions.
        """
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
        """Calculates the probability of y, P(y | a, x).

        Args:
            data: observation context, x_t
            treatment: selected action a_t

        Returns
            An array of probabilities for the different possible outcomes.
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
        """Updates the model based on new observation.

        Args:
            user: x_t
            action: a_t
            outcome: y_t
        """
        self.data = np.vstack((self.data, user))
        action_matrix = self.get_action_matrix(1, action)
        self.actions = np.vstack((self.actions, action_matrix))
        self.outcome = np.vstack((self.outcome, outcome))

        x = np.hstack((self.data, self.actions))
        self.model = self.model.fit(x, self.outcome.flatten())


@attr.s
class Approach1_adap_thomp(RecommenderModel):

    def fit_treatment_outcome(self, data, actions, outcomes):
        """Fits the model with the historical data. This is the base model where
        the adapative recommender will add observations sequentially.

        Args:
            data: the covariates for the users
            actions: the selected (historical) actions for the users
            outcomes: the observed (historical) outcomes for the users 
        """
        self.actions = actions
        self.outcome = outcomes
        self.data = data

        action_matrix = self.get_action_matrix(len(data), actions)
        self.actions = action_matrix
        x = np.hstack((data.copy(), action_matrix))

        alg3 = Algorithm3()
        alg3.initialize(lam=0.2, parameters=len(x[0]))
        alg3.fit(x, outcomes.flatten())
        self.model = alg3

    def get_action_probabilities(self, user_data):
        """Calculates the action probabilities based on one observation context
        x_t.

        Args: 
            user_data: context (x_t) for one observation

        Returns:
            A ndarray of length equal to n_actions.
        """
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
        """Predicts P(y|a, x).

        Args:
            data: x_t
            treatment: a_t
        
        Returns
            An array of probabilities conditioned on a, x.
        """
        treatment_vector = self.get_treatment_vector(treatment)

        if isinstance(data, pd.core.series.Series):
            x_t = data.to_numpy()
            user_array = np.hstack((x_t, treatment_vector)).reshape(1, -1)

        else:
            user_array = np.hstack((data.flatten(), treatment_vector)).reshape(1, -1)

        # P(y_t | a_t, x_t)
        p = self.model.predict_proba(user_array)
        return p

    def estimate_expected_conditional_reward(self, user_data, action):
        """Estimates the expected reward conditional on an action.

        Args:
            user_data: the covariates of an observation
            action: the selected action (a_t = a)

        Returns:
            The expected conditional reward E[r_t | a_t = a].
        """
        estimated_cond_utility = 0

        p_y = self.predict_proba(user_data.copy(), action)

        # sum over possible outcomes for expected reward
        for y_t in range(self.n_outcomes):
            estimated_cond_utility += p_y[y_t] * self.reward(action, y_t)

        return estimated_cond_utility

    def observe(self, user, action, outcome):
        """Updates the model based on new observation.

        Args:
            user: x_t
            action: a_t
            outcome: y_t
        """
        a_vector = self.get_treatment_vector(action)
        x = np.array([np.hstack((user, a_vector))])
        self.model.fit(x, outcome)


@attr.s
class Approach1_adap_thomp_explore(Approach1_adap_thomp):

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_action_probabilities(self, user_data):
        """Calculates the action probabilities based on one observation context
        x_t. Uses epsilon-based exploration to occasionally select a random 
        action.

        Args: 
            user_data: context (x_t) for one observation

        Returns:
            A ndarray of length equal to n_actions.
        """
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

        # epsilon-based exploration
        threshold = uniform.rvs(loc=0, scale=1, size=1)
        if threshold < self.epsilon:
            a_star = np.random.choice(self.n_actions)
        else:
            a_star = np.argmax(expected_reward)

        pi[a_star] = 1
        return pi


@attr.s
class Approach1_adap_thomp_eps_varsel(Approach1_adap_thomp_explore):

    def fit_treatment_outcome(self, data, actions, outcomes):
        """Fits the model with the historical data. This is the base model where
        the adapative recommender will add observations sequentially.

        Args:
            data: the covariates for the users
            actions: the selected (historical) actions for the users
            outcomes: the observed (historical) outcomes for the users 
        """
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
        self.actions = action_matrix

        x_selected = np.hstack((data[:, selected_var], action_matrix))

        alg3 = Algorithm3()
        alg3.initialize(lam=0.2, parameters=len(x_selected[0]))
        alg3.fit(x_selected, outcomes.flatten())
        self.model = alg3

    def predict_proba(self, data, treatment):
        """Predicts the probability P(y|a, x) given the selected variables.

        Args:
            data: x_t
            treatment: a_t
        
        Returns
            The probabilities for the different y.
        """
        treatment_vector = self.get_treatment_vector(treatment)
        if isinstance(data, pd.core.series.Series):
            x_t = data.to_numpy()
            flat_data = x_t.flatten()
        else:
            flat_data = data.flatten()
        
        user_array = np.hstack((flat_data[self.selected_variables], treatment_vector)).reshape(1, -1)

        # P(y_t | a_t, x_t)
        p = self.model.predict_proba(user_array)
        return p

    def observe(self, user, action, outcome):
        """Updates the model based on new observation.

        Args:
            user: x_t
            action: a_t
            outcome: y_t
        """
        a_vector = self.get_treatment_vector(action)
        x = np.array([np.hstack((user[self.selected_variables], a_vector))])
        self.model.fit(x, outcome)


class Algorithm3:
    """This is an implementation of algorithm 3 from Chapelle & Li. It is also
    adapted from the tutorial of 'gdmarmerola' (see the report).
    """

    def initialize(self, lam, parameters):
        """Initialize step in algorithm 3 from Chapelle and Li.

        Args:
            lam: the initial inverse precision for each of the regression 
                coefficients
            parameters: the number of covariates used (in case of variable 
                selection)
        """
        self.l = lam
        self.p = parameters
        self.m = np.zeros(parameters)
        self.q = np.repeat(lam, parameters)

        # initial regression coefficients
        self.w = norm.rvs(loc=self.m, scale=1/self.q, size=self.p)

    def argmin_w(self, w, x, y):
        """Second point in algorithm 3 from Chapelle and Li.

        Args:
            w: regression coefficients
            x: covariates for a specific observation
            y: the outcome

        Returns:
            The minimizing w.
        """
        if len(x) == 1:
            y = np.array([y])

        partial_sum = self.q * (w - self.m)
        obs_sum = 0
        for j in range(len(x)):
            obs_sum += np.log(1 + np.exp(-y[j] * w.dot(x[j])))

        w_temp = 0.5 * partial_sum.dot(w - self.m) + obs_sum

        return w_temp

    def gradient_w(self, w, x, y):
        """Gradient of w, calculated in the report.

        Args:
            w: regression coefficients
            x: context x_t
            y: outcome y_t

        Return
            The value of the gradient grad(x, y).
        """

        if len(x) == 1:
            y = np.array([y])

        partial_sum = 0
        for j in range(len(x)):
            partial_sum += (y[j]*x[j])/(1 + np.exp(-y[j] * w.dot(x[j])))

        grad = self.q * (w - self.m) - partial_sum
        return grad

    def fit(self, x, y):
        """Performs one iteration of the fitting in algorithm 3.

        Args:
            x: the covariates (x_t)
            y: the outcome (y_t)
        """
        # argmin w
        argmin_w_result = minimize(self.argmin_w, self.w, args=(
            x, y), jac=self.gradient_w, method="Newton-CG").x

        self.w = argmin_w_result
        self.m = argmin_w_result
        self.update_q(self.m, x)

    def update_q(self, w, x):
        """Performs the updating of q in algorithm 3.

        Args:
            w: weights to be used in the logistic regression
            x: context x_t
        """
        n = len(x)

        # Laplace approximation, vector of length n
        inner_product = x.dot(w)
        p = (1+np.exp(-inner_product))**(-1)

        # for each parameter
        for i in range(self.p):
            q = self.q[i]

            # for each observation
            for j in range(n):
                q += (x[j, i]**2)*(p[j]*(1-p[j]))

            # update variance
            self.q[i] = q

    def predict_proba(self, x):
        """Predict probability of x, that is, P(y | a, x).

        Args:
            x: the covariates of an observation (context)

        Returns
            The probabilities for y = 0, y = 1
        """
        self.w = norm.rvs(loc=self.m, scale=1/self.q, size=self.p)

        # logistic function
        prob1 = 1/(1 + np.exp(- self.w.dot(x.flatten())))

        prob_array = np.zeros(2)
        prob_array[0] = 1 - prob1
        prob_array[1] = prob1
        return prob_array


class AdaptiveRecommender:

    def __init__(self, n_actions, n_outcomes):
        """Constructor for AdaptiveRecommender.

        Args:
            n_actions: number of actions possible
            n_outcomes: number of outcomes possible
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    # By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        """Sets the default reward equal to the outcome.

        Args:
            action: a
            outcome: y
        
        Returns:
            Outcome (y).
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

    def fit_data(self, data):
        """Fits an unsupervised model to the data.

        Args:
            data: the observations (x_t)
        """
        print("Preprocessing data")
        return None

    def fit_treatment_outcome(self, data, actions, outcome, recommender_model=None):
        """Fits historical data to the policy. Includes the action as a 
        covariate in the logistic regression model.

        Args:
            data: covariates, x_t
            actions: the action taken for each observation
            outcome: the outcome y_t | a_t, x_t
        """
        print("- Start fitting treatment outcomes")

        x = data.copy()
        if isinstance(data, pd.DataFrame):
            x = x.to_numpy()

        if isinstance(actions, pd.DataFrame):
            actions = actions.to_numpy()

        if isinstance(outcome, pd.DataFrame):
            outcome = outcome.to_numpy()

        data = x
        x = np.hstack((x, actions))

        if recommender_model is None:
            recommender_model = Approach1_adap_bl(
                self.n_actions, self.n_outcomes)

        recommender_model.fit_treatment_outcome(data, actions, outcome)
        recommender_model.set_reward(self.reward)
        self.recommender_model = recommender_model
        print("- Stop fitting treatment outcomes")

    def estimate_utility(self, data, actions, outcome, policy=None, observe=False):
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

            # observe outcome
            if observe:
                self.observe(user_data, actions.iloc[t], outcome.iloc[t])

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

    def predict_proba(self, data, treatment):
        """Calculates the conditional probability P(y | a, x).

        Args:
            data: x
            treatment: a

        Returns
            Distribution over y.
        """
        return self.recommender_model.predict_proba(data, treatment)

    def get_action_probabilities(self, user_data):
        """Calculates the conditional distribution of actions pi(a_t | x_t).

        Args:
            user_data: observation to calculate the conditional distribution for   

        Returns
            Distribution over a.          
        """
        return self.recommender_model.get_action_probabilities(user_data)

    def recommend(self, user_data):
        """Recommends an action based on x_t.

        Args:
            user_data: x_t
        
        Returns
            An action a_t.
        """
        return np.random.choice(self.n_actions, p=self.get_action_probabilities(user_data))

    def observe(self, user, action, outcome):
        """Updates the model based on new observation.

        Args:
            user: x_t
            action: a_t
            outcome: y_t
        """
        self.recommender_model.observe(user, action, outcome)

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None


if __name__ == "__main__":
    np.random.seed(1)
    data = MedicalData()
    n_actions = len(np.unique(data.a_train))
    n_outcomes = len(np.unique(data.y_train))

    ######################
    # Logistic-Bernoulli #
    ######################

    """ ada_recommender = AdaptiveRecommender(n_actions, n_outcomes)
    ada_recommender.set_reward(lambda a, y: y - 0.1*(a != 0))
    ada_recommender.fit_treatment_outcome(
        data.x_train, data.a_train, data.y_train)

    ada_estimated_utility = ada_recommender.estimate_utility(
        data.x_test, data.a_test, data.y_test, observe=True)
    print(f"Estimated expected utility = {round(ada_estimated_utility, 4)}") """
   

    ###############################
    # Logistic regression with TS #
    ###############################

    """ ada_thomp_recommender = AdaptiveRecommender(n_actions, n_outcomes)
    ada_thomp_recommender.set_reward(lambda a, y: y - 0.1*(a != 0))

    thomp_policy = Approach1_adap_thomp(
        ada_thomp_recommender.n_actions, ada_thomp_recommender.n_outcomes)

    ada_thomp_recommender.fit_treatment_outcome(
        data.x_train, data.a_train, data.y_train, thomp_policy)

    ada_thomp_estimated_utility = ada_thomp_recommender.estimate_utility(
        data.x_test, data.a_test, data.y_test, observe=True)
    print(
        f"Estimated expected utility = {round(ada_thomp_estimated_utility, 4)}") """

    ###########################################
    # Logistic regression with TS and explore #
    ###########################################

    """ ada_ts_eps_recommender = AdaptiveRecommender(n_actions, n_outcomes)
    ada_ts_eps_recommender.set_reward(lambda a, y: y - 0.1*(a != 0))

    thomp_policy_explore = Approach1_adap_thomp_explore(
        ada_ts_eps_recommender.n_actions, ada_ts_eps_recommender.n_outcomes)
    thomp_policy_explore.set_epsilon(epsilon=0.10)

    ada_ts_eps_recommender.fit_treatment_outcome(
        data.x_train, data.a_train, data.y_train, thomp_policy_explore)

    ada_ts_eps_estimated_utility = ada_ts_eps_recommender.estimate_utility(
        data.x_test, data.a_test, data.y_test, observe=True)
    print(
        f"Estimated expected utility = {round(ada_ts_eps_estimated_utility, 4)}") """

    ###############################################################
    # Logistic regression with TS, explore and variable selection #
    ###############################################################

    ada_ts_eps_varsel_recommender = AdaptiveRecommender(n_actions, n_outcomes)
    ada_ts_eps_varsel_recommender.set_reward(lambda a, y: y - 0.1*(a != 0))

    thomp_policy_explore_varsel = Approach1_adap_thomp_eps_varsel(
        ada_ts_eps_varsel_recommender.n_actions, ada_ts_eps_varsel_recommender.n_outcomes)
    thomp_policy_explore_varsel.set_epsilon(epsilon=0.10)

    ada_ts_eps_varsel_recommender.fit_treatment_outcome(
        data.x_train, data.a_train, data.y_train, thomp_policy_explore_varsel)

    ada_ts_eps_varsel_estimated_utility = ada_ts_eps_varsel_recommender.estimate_utility(
        data.x_test, data.a_test, data.y_test, observe=True)
    print(
        f"Estimated expected utility = {round(ada_ts_eps_varsel_estimated_utility, 4)}")
