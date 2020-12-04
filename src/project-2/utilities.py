import attr
import numpy as np


@attr.s
class RecommenderModel:
    """Superclass for recommender models, contains common functionality for the
    recommender models.
    """

    n_actions = attr.ib()
    n_outcomes = attr.ib()

    def set_reward(self, reward):
        self.reward = reward


@attr.s
class FixedTreatmentPolicy:
    """Represents fixed treatment policies.

    """

    n_actions = attr.ib()
    # the specific treatment
    fixed_action = attr.ib()

    def get_action_probabilities(self, user_data):
        """Calculates action probabilities for a fixed treatment policy. This
        will always be 100 % for the fixed treatment (a_t).

        Args:
            user_data: the observation (x_t)

        Returns:
            The distribution pi(a|x).
        """
        pi = np.zeros(self.n_actions)
        pi[self.fixed_action] = 1
        return pi

    def recommend(self):
        return self.fixed_action


class FinalAnalysis:

    def fixed_treatment_policy_check(self, recommender, n_tests=1000, generator=None):
        """Calculates the expected utility for the possible fixed treatment 
        policies and returns the expected utility.

        Args:
            recommender: a recommender (historical, improved or adaptive)
            n_tests: the number of observations to check
            generator: whether or not to calculate expected utility, if False,
                calculate using the generator
        """

        action_utilities = np.zeros((recommender.n_actions, 2))

        if generator is None:
            data = recommender.recommender_model.data[-n_tests:, ]
            actions = recommender.recommender_model.actions[-n_tests:, ]
            outcomes = recommender.recommender_model.outcome[-n_tests:, ]

            for a_t in range(recommender.n_actions):
                fixed_policy = FixedTreatmentPolicy(recommender.n_actions, a_t)
                action_utilities[a_t, 0] = a_t
                action_utilities[a_t, 1] = recommender.estimate_utility(
                    data, actions, outcomes, fixed_policy)
        else:
            print("Generating actual outcomes")
            for a_t in range(recommender.n_actions):
                print(f"a_t = {a_t}")
                fixed_policy = FixedTreatmentPolicy(recommender.n_actions, a_t)
                rewards = np.zeros(n_tests)

                for t in range(n_tests):
                    x = generator.generate_features()
                    a = fixed_policy.recommend()
                    y = generator.generate_outcome(x, a)
                    rewards[t] = recommender.reward(a, y)

                action_utilities[a_t, 1] = np.mean(rewards)

        return action_utilities