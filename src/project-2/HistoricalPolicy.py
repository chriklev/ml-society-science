import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


class HistoricalPolicy:

    def __init__(self, n_actions, n_outcomes, actions, outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.actions = actions
        self.outcomes = outcomes

        self._calculate_pi0(actions)
        self._calculate_theta(actions, outcomes)

    def _calculate_pi0(self, actions):
        """Calculates pi0_hat from the historical actions.

        Args:
            actions: historical actions
        """
        pi0_hat_a1 = np.sum(actions.to_numpy())/len(actions)
        pi0_hat_a0 = 1 - pi0_hat_a1
        self.pi0_hat = (pi0_hat_a0, pi0_hat_a1)

    def _calculate_theta(self, actions, outcomes):
        """Calculates theta_a_hat from the actions and the outcomes.

        Args:
            actions: the historical actions
            outcomes: the historical outcomes
        """
        a0_y1 = outcomes[actions["action"] == 0].to_numpy()
        a0 = actions[actions["action"] == 0].to_numpy()

        a1_y1 = outcomes[actions["action"] == 1].to_numpy()
        a1 = actions[actions["action"] == 1].to_numpy()

        theta0 = np.sum(a0_y1)/len(a0)
        theta1 = np.sum(a1_y1)/len(a1)
        self.theta_hat = (theta0, theta1)

    def sample(self, n_samples):
        """Samples from the historical policy pi0.

        Args:
            n_samples: the number of samples

        Returns
            The samples.
        """
        tfd = tfp.distributions

        model = tfd.JointDistributionNamedAutoBatched({
            'a': tfd.Independent(
                tfd.Bernoulli(probs=self.pi0_hat[1])),
            'y': lambda a:
            tfd.Independent(
                tfd.Bernoulli(probs=self.theta_hat[a]))
        })

        samples = [model.sample() for _ in range(n_samples)]

        a_samples = [samples[_]['a'].numpy() for _ in range(len(samples))]
        y_samples = [samples[_]['y'].numpy() for _ in range(len(samples))]

        df = pd.DataFrame({'actions': a_samples, 'outcomes': y_samples})
        return df

    def estimate_expected_utility(self, rep, n):
        """Calculates the expected utility.

        Args:
            rep: number of repetitions
            n: number of observations to sample
        """

        # number of sample observations
        #n = len(self.actions)

        # expected utilities
        expected_U = np.empty(shape=(rep, n))

        for r in range(rep):
            # store expected utility for each observation
            expected_utility = np.zeros(n)

            # sample data from model 0
            sample_data = self.sample(n)

            actions = sample_data['actions']
            outcomes = sample_data['outcomes']

            for i in range(n):
                expected_utility[i] = self.u(actions[i], outcomes[i])

            expected_U[r, :] = expected_utility

        return expected_U

    def u(self, a, y):
        """Calculates utility.

        Args:
            a: action
            y: outcome
        Returns:
            The utility
        """
        return -0.1*a + y

    def method0(self, rep, n):
        """Calculates expected utility and error bounds related to model 0.

        NOTE: this method is adapted from https://github.com/dhesse/IN-STK5000
        -Notebooks-2020/blob/master/notebooks/Lecture%2010%20-%20Multilevel%20
        Models.ipynb

        Args:
            rep: number of repitions to use
            n: number of samples in each repetition
        """
        expected_utilities = self.estimate_expected_utility(rep, n)

        confidence_interval = np.percentile(
            expected_utilities, [2.5, 97.5], axis=1)
        mean = np.mean(expected_utilities, axis=1)

        plt.errorbar(mean, range(len(mean)), xerr=(
            mean - confidence_interval[0], confidence_interval[1] - mean), marker='o', ls="")
        # plt.show()
        plt.savefig("img/part2_1_method0.png")

    def get_action_probabilities(self):
        """Gets the probabilities for the different actions.

        Returns:
            The probabilities for the different actions.
        """
        return self.theta_hat
