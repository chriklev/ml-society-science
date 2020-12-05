import martin_historical_recommender
import martin_improved_recommender
import martin_adaptive_recommender
import pandas
import data_generation
import numpy as np

features = pandas.read_csv(
    'data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv(
    'data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv(
    'data/medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:, 128] + features[:, 129]*2


def default_reward_function(action, outcome):
    return -0.1 * (action != 0) + outcome


def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        print(f"{t}/{T}") if (t % 100) == 0 else None
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        # print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u


def two_action_test(policy, recommender_model=None, n_tests=1000):
    """Tests two-action case from the TestRecommender class.

    Args:
        policy: a recommender class
        recommender_model: the internal model to use
    """
    # First test with the same number of treatments
    print("---- Testing with only two treatments ----")

    print("Setting up simulator")
    generator = data_generation.DataGenerator(
        matrices="./generating_matrices.mat")
    # Fit the policy on historical data first
    print("Fitting historical data to the policy")
    if recommender_model is None:
        policy.fit_treatment_outcome(features, actions, outcome)
    else:
        policy.fit_treatment_outcome(
            features, actions, outcome, recommender_model)
    # Run an online test with a small number of actions
    print(f"Running an online test ({n_tests})")
    result = test_policy(generator, policy, default_reward_function, n_tests)
    print("Total reward:", result)
    print("Final analysis of results")
    # policy.final_analysis()


def multiple_action_test(policy, recommender_model=None, n_tests=1000):
    print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
    print("Setting up simulator")
    generator = data_generation.DataGenerator(
        matrices="./big_generating_matrices.mat")
    # Fit the policy on historical data first
    print("Fitting historical data to the policy")
    if recommender_model is None:
        policy.fit_treatment_outcome(features, actions, outcome)
    else:
        policy.fit_treatment_outcome(
            features, actions, outcome, recommender_model)
    # Run an online test with a small number of actions
    print(f"Running an online test ({n_tests})")
    result = test_policy(generator, policy, default_reward_function, n_tests)
    print("Total reward:", result)
    print("Final analysis of results")
    # policy.final_analysis()


if __name__ == "__main__":
    np.random.seed(1)
    ##############
    # Historical #
    ##############

    """ hist_recommender2 = martin_historical_recommender.HistoricalRecommender(
        2, 2)
    two_action_test(hist_recommender2) 

    hist_recommender129 = martin_historical_recommender.HistoricalRecommender(
        129, 2)
    multiple_action_test(hist_recommender129) """

    ############
    # Improved #
    ############

    """ imp_recommender2 = martin_improved_recommender.ImprovedRecommender(2, 2)
    two_action_test(imp_recommender2) 

    imp_recommender129 = martin_improved_recommender.ImprovedRecommender(
        129, 2)
    multiple_action_test(imp_recommender129) """

    ############
    # Adaptive #
    ############

    """ ada_recommender2 = martin_adaptive_recommender.AdaptiveRecommender(2, 2)
    policy_model2 = martin_adaptive_recommender.Approach1_adap_thomp_explore(
        2, 2)
    policy_model2.set_epsilon(0.10)
    two_action_test(ada_recommender2, recommender_model=policy_model2) """

    ada_recommender129 = martin_adaptive_recommender.AdaptiveRecommender(
        129, 2)
    policy_model129 = martin_adaptive_recommender.Approach1_adap_thomp_explore(
        129, 2)
    policy_model129.set_epsilon(0.05)
    multiple_action_test(ada_recommender129,
                         recommender_model=policy_model129, n_tests=2000)

    """ ada_varsel_recommender129 = martin_adaptive_recommender.AdaptiveRecommender(
        129, 2)
    policy_model_varsel129 = martin_adaptive_recommender.Approach1_adap_thomp_eps_varsel(
        129, 2)
    policy_model_varsel129.set_epsilon(0.05)
    multiple_action_test(ada_varsel_recommender129,
                         recommender_model=policy_model_varsel129, n_tests=2000) """
