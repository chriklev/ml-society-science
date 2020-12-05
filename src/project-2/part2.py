from part1 import MedicalData
from random_recommender import RandomRecommender
from part2_historical_recommender import HistoricalRecommender
from part2_improved_recommender import ImprovedRecommender, Approach1_policy
from HistoricalPolicy import HistoricalPolicy
import pandas as pd
import numpy as np

if __name__ == "__main__":
    np.random.seed(1)
    data = MedicalData()

    x_joined = [data.x_train, data.x_test]
    x = pd.concat(x_joined)
    a_joined = [data.a_train, data.a_test]
    a = pd.concat(a_joined)
    y_joined = [data.y_train, data.y_test]
    y = pd.concat(y_joined)

    # exercise 1
    hr = HistoricalRecommender(2, 2)
    uh = hr.estimate_utility(x, a, y)
    print(f"Average utility = {round(uh, 4)}")

    hp = HistoricalPolicy(2, 2, a, y)
    hp.method0(100, 1000, 5, 0.05)

    hr.estimate_utility(x, a, y, hp)
    print(f"pi0_hat = {hp.pi0_hat}")
    print(f"theta_0 = {hp.theta_hat[0]}")
    print(f"theta_1 = {hp.theta_hat[1]}")

    # bootstrap
    boot_util = hp.bootstrap_expected_utility(500)
    hp.plot_bootstrap_hist(boot_util, 500)
    hp.bootstrap_percentile(100, 5, 0.05)
    hp.plot_bootstrap_ci(100, 5)

    # exercise 2
    model_data = x.assign(a=a, y=y)

    improved = ImprovedRecommender(2, 2)
    improved.set_reward(lambda a, y: y - 0.1*(a != 0))

    # sub policy 1
    approach1 = Approach1_policy(2, 2)
    approach1.fit_data(model_data, var_sel=True)

    im_util = improved.estimate_utility(
        data.x_test, data.a_test, data.y_test, approach1)
    print(f"Expected utility = {round(im_util, 4)}")
