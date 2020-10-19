import TestImplementation
import group1_banker
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def countplot():
    # Get data
    X = TestImplementation.get_data()
    y = X.pop("repaid")

    # Fit the banker
    banker = group1_banker.Group1Banker()
    banker.set_interest_rate(.05)
    banker.fit(X, y)

    # Get predictions
    n_rows = X.shape[0]
    y_predicted = np.empty(n_rows)
    for i in range(n_rows):
        y_predicted[i] = banker.get_best_action(X.iloc[i])

    is_female = X["marital status_A92"] == 1
    sex = pd.Series(is_female.map({True: "female", False: "male"}))

    gender_data = pd.DataFrame()
    gender_data["repaid"] = pd.concat((y, pd.Series(y_predicted))).map(
        {0: "no", 1: "yes"})
    gender_data["response"] = np.repeat(["true", "predicted"], y.size)
    gender_data["sex"] = pd.concat((sex, sex))

    sns.set_style(style="whitegrid")
    g = sns.catplot(x="repaid", hue="sex", col="response",
                    data=gender_data, kind="count",
                    height=4, aspect=.7)
    plt.savefig("img/gender_compare.png")
    plt.show()


if __name__ == "__main__":
    countplot()
