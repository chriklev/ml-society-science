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

        def find_probs(a):
            probs = np.zeros(len(a.shape))
            print(type(a))
            for i in range(len(a.shape)):
                print(f"type{type(a[i])}")
                print(f"len{a[i].shape}")
                print(f"{tf.cast(a[i], tf.int32)}")
                probs[i] = self.theta_hat[int(tf.cast(a[i], tf.int32))]
            return probs

        model = tfd.JointDistributionNamedAutoBatched({
            'a':
            tfd.Independent(
                tfd.Bernoulli(
                    probs=self.pi0_hat[1][..., tf.newaxis]), reinterpreted_batch_ndims=1
            ),
            'y': lambda a:
                tfd.Independent(tfd.Bernoulli(
                    probs=find_probs(a)), reinterpreted_batch_ndims=1)
        })

        # def log_prob(a):
        #     return model.log_prob({'a': a, 'y': tfd.Bernoulli(probs=self.theta_hat[a])})

        # num_res = 100
        # num_burnin = 100
        # num_chains = 4
        # breakpoint()

        # @ tf.function
        # def sample(num_res, num_burnin, num_chains):
        #     first_sample = model.sample(num_chains)
        #     kernel = tfp.mcmc.HamiltonianMonteCarlo(
        #         target_log_prob_fn=log_prob,
        #         num_leapfrog_steps=4,
        #         step_size=0.2
        #     )
        #     states = tfp.mcmc.sample_chain(
        #         kernel=kernel,
        #         num_burnin_steps=num_burnin,
        #         num_results=num_res,
        #         current_state=[first_sample['a']]
        #     )
        #     return states

        # test = sample(num_res, num_burnin, num_chains)

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
        # n = len(self.actions)

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

    def method0(self, rep, n, repeats, alpha):
        """Calculates expected utility and error bounds related to model 0.

        NOTE: this method is adapted from https://github.com/dhesse/IN-STK5000
        -Notebooks-2020/blob/master/notebooks/Lecture%2010%20-%20Multilevel%20
        Models.ipynb

        Args:
            rep: number of repitions to use
            n: number of samples in each repetition
            repeats: number of confidence intervals
            alpha: confidence level
        """
        repetitions = np.zeros(repeats)
        percentile_ci = np.zeros(shape=(2, repeats))

        sample_means = np.zeros(shape=(repeats, rep))

        mean_values = list()

        lower_ci = int(round(np.ceil((alpha*(rep+1)/2))))
        upper_ci = rep - lower_ci
        middle_idx = round(rep/2)

        for i in range(repeats):
            expected_utilities = self.estimate_expected_utility(rep, n)
            mean = np.mean(expected_utilities, axis=1)

            sorted_mean = np.sort(mean)
            repetitions[i] = sorted_mean[middle_idx]
            percentile_ci[0, i] = sorted_mean[lower_ci]
            percentile_ci[1, i] = sorted_mean[upper_ci]

            mean_values.extend(mean)
            sample_means[i, ] = mean

        # self.plot_percentile_interval(repetitions, percentile_ci, rep)
        # self.plot_expected_frequency(mean_values, rep)
        self.bootstrap_ci(sample_means, alpha)

    def bootstrap_ci(self, sample_means):
        """Calculates bootstrap ci from sample means

        Args:
            sample_means: the sample means of the expected utility
        """
        n_repeats = len(sample_means)
        n_b = len(sample_means[0])
        means = np.mean(sample_means, axis=1)
        empirical_var = np.zeros(n_repeats)
        boot_mean = np.mean(means)

        for rep in range(n_repeats):
            empirical_var[rep] = np.sum((means[rep] - boot_mean)**2)/(n_b - 1)

        plt.clf()
        plt.errorbar(means, range(len(means)), xerr=1.96 *
                     np.sqrt(empirical_var), marker='o', ls="")
        plt.title("Bootstrap CI")
        plt.xlabel("E[U]")
        plt.ylabel("samples")
        plt.savefig("img/part2_1_method0_bootci.png")

    def plot_percentile_interval(self, repetitions, percentile_ci, rep):

        plt.errorbar(repetitions, range(len(repetitions)), xerr=(
            repetitions - percentile_ci[0], percentile_ci[1] - repetitions), marker='o', ls="")
        plt.title(f"Error bounds for expected utility ({rep} samples)")
        plt.xlabel("E[U]")
        plt.ylabel("samples")
        plt.savefig("img/part2_1_method0_error.png")
        plt.clf()

    def plot_expected_frequency(self, mean_values, rep):
        plt.hist(mean_values)
        plt.xlabel("E[U]")
        plt.ylabel("frequency")
        plt.title(f"Histogram of expected values ({rep} samples)")
        plt.savefig("img/part2_1_method0_hist.png")

    def get_action_probabilities(self):
        """Gets the probabilities for the different actions.

        Returns:
            The probabilities for the different actions.
        """
        return self.theta_hat
