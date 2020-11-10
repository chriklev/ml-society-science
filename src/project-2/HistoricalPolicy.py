import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf


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
        self.pi0_hat = np.sum(actions.to_numpy())/len(actions)

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
                tfd.Bernoulli(probs=self.pi0_hat)),
            'y': lambda a:
            tfd.Independent(
                tfd.Bernoulli(probs=self.theta_hat[a]))
        })

        samples = [model.sample() for _ in range(n_samples)]

        a_samples = [samples[_]['a'].numpy() for _ in range(len(samples))]
        y_samples = [samples[_]['y'].numpy() for _ in range(len(samples))]
        breakpoint()
        pass

    def get_action_probabilities(self):
        """Gets the probabilities for the different actions.

        Returns:
            The probabilities for the different actions.
        """
        return self.theta_hat
