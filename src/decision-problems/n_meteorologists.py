# Skeleton code

# Assume the number of models is equal to n=len(prior).  The argument
# P is an n-by-m array, where m is the number of possible
# predictions so that P[i][j] is the probability the i-th model assignsm to the j-th outcome. The outcome is a single number in 1, m.
import numpy as np
import random

## Calculate the posterior given a prior belief, a set of predictions, an outcome
## - prior: belief vector so that prior[i] is the probabiltiy of mdoel i being correct
## - P: P[i][j] is the probability the i-th model assignsm to the j-th outcome
## - outcome: actual outcome
def get_posterior(prior, P, outcome):
    n_models = len(prior)
    posterior = prior
    for k in range(n_models):
        posterior[k] *= P[k][outcome]
    posterior /= sum(posterior)
    ## So probability of outcome for model i is just...
    ## FILL IN
    return posterior


## Get the probability of the specific outcome given your current
## - belief: vector so that belief[i] is the probabiltiy of mdoel i being correct
## - P: P[i][j] is the probability the i-th model assignsm to the j-th outcome
## - outcome: actual outcome
def get_marginal_prediction(belief, P, outcome):
    n_models = len(belief)
    outcome_probability = 0
    for k in range(n_models):
        outcome_probability += belief[k] * P[k][outcome]
    return outcome_probability

## In this function, U[action,outcome] should be the utility of the action/outcome pair
def get_expected_utility(belief, P, action, U):
    n_models = len(belief)
    n_outcomes = np.shape(P)[1]
    utility = 0 ## FILL IN
    for x in range(n_outcomes):
        P_x = get_marginal_prediction(belief, P, x)
        utility += P_x * U[action,x]
    return utility

## In this function, U[action,outcome] should be the utility of the action/outcome pair, using MAP inference
def get_MAP_utility(belief, P, action, U):
    n_models = len(belief)
    n_outcomes = np.shape(P)[1]
    utility = 0 ## FILL IN
    MAP_model = belief.argmax() # get the maximising model
    for x in range(n_outcomes):
        P_x = P[MAP_model][x]
        utility += P_x * U[action,x]
    return utility


## Here you should return the action maximising expected utility
## Here we are using the Bayesian marginal prediction
def get_best_action(belief, P, U):
    n_models = len(belief)
    n_actions = np.shape(U)[0]
    util = np.zeros(n_actions)
    for a in range(n_actions):
        util[a] = get_expected_utility(belief, P, a, U)
    best_action = np.argmax(util)
    return best_action

## Here you should return the action maximising expected utility
## Here we are using the MAP model
def get_best_action_MAP(belief, P, U):
    n_models = len(belief)
    n_actions = np.shape(U)[0]
    util = np.zeros(n_actions)
    for a in range(n_actions):
        util[a] = get_MAP_utility(belief, P, a, U)
    best_action = np.argmax(util)
    return best_action



T = 4 # number of time steps
n_models = 3 # number of models

# build predictions for each station of rain probability
prediction = np.matrix('0.1 0.1 0.3 0.4; 0.4 0.1 0.6 0.7; 0.7 0.8 0.9 0.99')


n_outcomes = 2 # 0 = no rain, 1 = rain

## we use this matrix to fill in the predictions of stations
P = np.zeros([n_models, n_outcomes])
belief = np.ones(n_models) / n_models;
rain = [1, 0, 1, 1];

print("a x U")
print("-----")
for t in range(T):
    # utility to loop to fill in predictions for that day
    for model in range(n_models):
        P[model,1] = prediction[model,t] # the table predictions give rain probabilities
        P[model,0] = 1.0 - prediction[model,t] # so no-rain probability is 1 - that.
    probability_of_rain = get_marginal_prediction(belief, P, 1)
    #print(probability_of_rain)
    U  = np.matrix('1 -10; 0 0')
    action = get_best_action_MAP(belief, P, U)
    
    print(action, rain[t], U[action, rain[t]])
    belief = get_posterior(belief, P, rain[t])
    print(belief)

                
    
