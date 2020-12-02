#import adaptive_recommender
#import historical_recommender
#import improved_recommender
import random_recommender
import logistic_recommender
import data_generation
import numpy as np
import pandas


def default_reward_function(action, outcome):
    return -0.1 * (action != 0) + outcome


def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        # print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
        print(f"{t}/{T}")
    return u


features = pandas.read_csv(
    'data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv(
    'data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv(
    'data/medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:, 128] + features[:, 129]*2

#policy_factory = random_recommender.RandomRecommender

#policy_factory = historical_recommender.HistoricalRecommender

#policy_factory = adaptive_recommender.AdaptiveRecommender

# policy_factory = improved_recommender.ImprovedRecommender

policy_factory = logistic_recommender.LogisticRecommender

#import reference_recommender
#policy_factory = reference_recommender.HistoricalRecommender

# First test with the same number of treatments
print("---- Testing with only two treatments ----")

print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
print(policy.n_actions, policy.n_outcomes)
# Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
# Run an online test with a small number of actions
print("Running an online test")
n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward:", result)
print("Final analysis of results")
policy.final_analysis()

# First test with the same number of treatments
print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
print("Setting up simulator")
generator = data_generation.DataGenerator(
    matrices="./big_generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
# Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
# Run an online test with a small number of actions
print("Running an online test")
n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward:", result)
print("Final analysis of results")
policy.final_analysis()
