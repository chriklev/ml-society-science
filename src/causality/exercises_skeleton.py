from exercises import *

import numpy as np

model = BasicModel(0.3)
policy = BasicPolicy(0.5)

n_data = 1000

a = np.zeros(n_data)
x = np.zeros(n_data)
for t in range(n_data):
    a[t] = policy.get_action()
    x[t] = model.get_response(a[t])
    

y = np.zeros(n_data)
for t in range(n_data):
    y[t] = x[t] - a[t]

print("Average outcome:", np.mean(y))

### Get an estimate for theta for the model where the response is one of two different normal distributions, whose mean arbitrarily depends on the action
#theta = 0.1*np.random.normal(size=2)
theta = 0.1
model = BasicModel(theta)
policy = BasicPolicy(0.1)
n_samples = 1000
n_actions = policy.get_n_actions()
a = np.empty(n_samples, dtype=int)
y = np.zeros(n_samples)
#hat_theta = np.zeros(2) # use this to estimate the model
hat_theta = 0
hat_pi = np.zeros(2) # use this to estimate the policy
counts = np.zeros(2) # this is needed for getting the right
hat_U = 0


## Generate data
for t in range(n_samples):
    a[t] = int(policy.get_action()) 
    y[t] = model.get_response(a[t])

for t in range(n_samples):
    hat_U += y[t]
    hat_theta += y[t] - a[t]
    counts[a[t]] += 1
    
hat_U /= n_samples
hat_theta /= n_samples
hat_pol = counts / counts.sum()
print("Actual parameter: ", theta)
print("Estimated parameter:", hat_theta)
print("Estimated utility:", hat_U)
print("Estimated policy:", hat_pol)

## Exercise 1: Calculate the actual value of the improved policy by generating new data from that policy

## Exercise 2: Use the covariate model to generate data, and then obtain a new policy that is optimal for this covariate model






