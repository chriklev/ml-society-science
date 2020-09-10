import numpy as np
from sklearn.linear_model import LogisticRegression

class NameBanker:

    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        """Fits the logistic regression model.

        Args:
            X:
                The covariates of the data set.
            y:
                The response variable from the data set.
        """
        self.data = [X, y]
        log_reg_object = LogisticRegression(random_state=1)
        self.model = log_reg_object.fit(X, y)

    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        """Predicts the probability of a new observation given the model.

        Args:
            x:
                A new, independent observation.
        """
        return self.model.predict_proba(x)

    def get_proba(self):
        return np.random.uniform(0, 1)

    # The expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
        """Calculate expected utility.
        
        """
        if action == 0:
            return 0
        
        r = self.rate
        p_c = self.get_proba()
        n = x.length_of_loan
        m = x.amount_of_loan

        e_x = p_c * m * ((1 + r) ** n - 1) + (1 - p_c) * (-m)
        return e_x

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    def get_best_action(self, x):
        return np.random.choice(2, 1)[0]
