#import reference_recommender
import random_recommender
import data_generation
import numpy as np
import pandas
import json
from tqdm import tqdm
from martin_adaptive_recommender import AdaptiveRecommender, Algorithm3, Approach1_adap_thomp, Approach1_adap_thomp_explore
from martin_improved_recommender import Approach1_impr_varsel, Approach1_impr_bl, ImprovedRecommender
from martin_historical_recommender import HistoricalRecommender

from chris_adaptive_recommender import AdaptiveLogisticRecommender
from chris_improved_recommender import LogisticRecommender

from utilities import FixedTreatmentPolicy


def default_reward_function(action, outcome):
    return -0.1 * (action != 0) + outcome


def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = [0]*T
    for t in tqdm(range(T)):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u[t] = r
        policy.observe(x, a, y)
        # print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u


features = pandas.read_csv(
    'data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv(
    'data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv(
    'data/medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:, 128] + features[:, 129]*2

def fixed_treatments(n_tests = 5000, generator =data_generation.DataGenerator(
    matrices="./big_generating_matrices.mat")):
    n_actions = generator.get_n_actions()
    n_outcomes = generator.get_n_outcomes()
    personal_columns = ["sex", "smoker"]
    gene_columns = ["gene " + str(i) for i in range(1, 127)]
    symptom_columns = ["symptom 1", "symptom 2"]
    model = HistoricalRecommender(n_actions=n_actions, n_outcomes=n_outcomes)
    model.set_reward(lambda a, y: y - 0.1*(a != 0))
    utilities = {}
    for a_t in tqdm(range(n_actions)):
        print(f"a_t = {a_t}")
        fixed_policy = FixedTreatmentPolicy(n_actions, a_t)
        rewards = [0]*n_tests
        covars = [0]*n_tests

        #Possibly redo the fixed treatments section
        for t in range(n_tests):
            x = generator.generate_features()
            a = fixed_policy.recommend()
            y = generator.generate_outcome(x, a)
            rewards[t] = model.reward(a, y)
            covars[t] = list(x) + [y]

        covars = pandas.DataFrame(covars, columns = personal_columns+gene_columns+symptom_columns+["y"])

        utilities["fixed_policy_"+str(a_t)] = [rewards,covars]
     
    
    return utilities


def final_full_analysis(n_tests = 1000, generator = data_generation.DataGenerator(
    matrices="./big_generating_matrices.mat")):
    """
    Generates utility measures for all models mentioned in the 
    paper except for those with exploration. 

    Args:
        n_tests: the number of online policy iterations to be passed to Christos' test policy function.
        generator: the generator for data. Determines the number of available actions.
    """
    print("START")
    n_actions = generator.get_n_actions()
    n_outcomes = generator.get_n_outcomes()
    model_list = [AdaptiveRecommender(n_actions=n_actions, n_outcomes=n_outcomes),
    AdaptiveRecommender(n_actions=n_actions, n_outcomes=n_outcomes),
    AdaptiveLogisticRecommender(n_actions=n_actions, n_outcomes=n_outcomes),
    ImprovedRecommender(n_actions=n_actions, n_outcomes=n_outcomes),
    ImprovedRecommender(n_actions=n_actions, n_outcomes=n_outcomes),
    LogisticRecommender(n_actions=n_actions, n_outcomes=n_outcomes),
    HistoricalRecommender(n_actions=n_actions, n_outcomes=n_outcomes)]
    model_names = ["adaptive_m_bl", 
    "adaptive_m_thomp",
    "adaptive_c", 
    "improved_m_bl",
    "improved_m_impr_varsel", 
    "improved_c",
     "historical"]
    utilities = {}
    print("N_ACTIONS ", n_actions)
    for model, name, in zip(model_list, model_names):
        print("On model ", name)
        if name[-1]=="p":
            model.fit_treatment_outcome(features, actions, outcome, Approach1_adap_thomp(n_actions, n_outcomes))
            model.set_reward(lambda a, y: y - 0.1*(a != 0))
        elif name[-1] == "l":
            model.fit_treatment_outcome(features, actions, outcome, Approach1_impr_varsel(n_actions, n_outcomes))
            model.set_reward(lambda a, y: y - 0.1*(a != 0))
        else:
            model.fit_treatment_outcome(features, actions, outcome)
            model.set_reward(lambda a, y: y - 0.1*(a != 0))
        results = test_policy(generator, model, default_reward_function, n_tests)
        utilities[name] = results
    
    #Fixed treatments
    for a_t in range(n_actions):
        print(f"a_t = {a_t}")
        fixed_policy = FixedTreatmentPolicy(n_actions, a_t)
        rewards = [0]*(n_tests)

        for t in range(n_tests):
            x = generator.generate_features()
            a = fixed_policy.recommend()
            y = generator.generate_outcome(x, a)
            rewards[t] = model_list[0].reward(a, y)

        utilities["fixed_policy_"+str(a_t)] = rewards
    return utilities


def test_exploration(n_tests = 1000, generator = data_generation.DataGenerator(
    matrices="./big_generating_matrices.mat"), epsilons = 10):
    """Generates utility measures for algorithms with exploration at various epsilon values.

    Args:
        n_tests: the number of online policy iterations to be passed to Christos' test policy function.
        generator: the generator for data. Determines the number of available actions.
        epsilons: 1/2 the number of epsilons at which the algorithm will be evaluated.
    """
    n_actions = generator.get_n_actions()
    n_outcomes = generator.get_n_outcomes()
    

    utilities ={}
    for epsilon in np.linspace(0, 0.5, epsilons):
        print("Epsilon = ", epsilon)
        ada_ts_eps_recommender = AdaptiveRecommender(n_actions, n_outcomes)
        ada_ts_eps_recommender.set_reward(lambda a, y: y - 0.1*(a != 0))
        thomp_policy_explore = Approach1_adap_thomp_explore(
        ada_ts_eps_recommender.n_actions, ada_ts_eps_recommender.n_outcomes)
        thomp_policy_explore.set_epsilon(epsilon=epsilon)
        ada_ts_eps_recommender.fit_treatment_outcome(features, actions, outcome, thomp_policy_explore)
        results = test_policy(generator, ada_ts_eps_recommender, default_reward_function, n_tests)
        utilities[epsilon] = results
    return utilities



#final_anal_dict = final_full_analysis()
#print("FINAL UTILITIES", final_anal_dict)
#print("DONE WITH TEST")
#with open('/Users/mjdioli/Documents/STK-IN5000/ml-society-science/src/project-2/final_analysis.json', 'w') as fp:
    #json.dump(final_anal_dict, fp)


#policy_factory = random_recommender.RandomRecommender
#policy_factory = reference_recommender.HistoricalRecommender

"""
# First test with the same number of treatments
print("---- Testing with only two treatments ----")

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
#print("Total reward:", result)
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

"""

