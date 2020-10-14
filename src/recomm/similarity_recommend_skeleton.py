## Generate data first

import numpy as np

## Calculate the distance between two people
## (helper function, if necessary)
def distance(x, y):
    return

## Get the similarity between two people.
## I recommend that this actually retunrs a similarity vector that sums to 1, and has a zero value for the user himself
def get_similarity(data, u, m):
    return similarity / np.sum(similarity)

## data[u,m] = 0 if somebody hasn't watched a movie
def infer_ratings(data):
    n_users = data.shape[0]
    n_movies = data.shape[1]
    inferred_ratings = data.copy()
    ## Do a loop wher you fill in values for each user and movie
    return inferred_ratings


    
        
def generate_random_data(n_users, n_movies, gamma):
    data = np.zeros([n_users, n_movies])
    rating_dist = [0.1, 0.2, 0.3, 0.3, 0.1]
    for u in range(n_users):
        n_ratings = 3; #np.random.geometric(gamma)
        ratings = np.random.permutation(range(n_movies))[0:n_ratings]
        for m in ratings:
            data[u,m] = 1 + np.random.choice(5,1, p=rating_dist)
    return data

data = generate_random_data(4, 5, 0.1)

inferred_data = infer_ratings(data)

print(data)
print(inferred_data)

