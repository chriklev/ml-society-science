import numpy as np
from sklearn.linear_model import LogisticRegression

class NameBanker:

    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        """Fits a logistic regression model.

        Args:
            X: The covariates of the data set.
            y: The response variable from the data set.
        """
        self.data = [X, y]

        log_reg_object = LogisticRegression(random_state=1, max_iter=2000)
        self.model = log_reg_object.fit(X, y)

    # set the interest rate
    def set_interest_rate(self, rate):
        """Sets the interest rate for the decision maker.

        Args:
            rate: the interest rate to use.
        """
        self.rate = rate
        return

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        """Predicts the probability for [0,1] given a new observation given the 
        model.

        Args:
            x: A new, independent observation.
        Returns:
            The prediction for class 1 given as the second element in the
            probability array returned from the model.
        """
        x = self._reshape(x)
        return self.model.predict_proba(x)[0][1]

    def get_proba(self):
        """Calculates probability of being credit-worthy.

        Returns:
            A float representing the probability of being credit-worthy.
        """
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

        Args:
            x: A new observation.
            action: Whether or not to grant the loan.
        """
        if action == 0:
            return 0

        r = self.rate
        p_c = self.predict_proba(x)

        # duration in months
        n = x['duration']
        # amount
        m = x['amount']

        e_x = p_c * m * ((1 + r) ** n - 1) + (1 - p_c) * (-m)
        return e_x

    def _reshape(self, x):
        """Reshapes Pandas Seris to a row vector.

        Args:
            x: Pandas Series.

        Returns:
            A ndarray as a row vector.
        """
        return x.values.reshape((1, len(x)))

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    def get_best_action(self, x):
        """Gets the best action defined as the action that maximizes utility.

        Args:
            x: A new observation.
        Returns:
            Best action based on maximizing utility.
        """
        expected_utility_give_loan = self.expected_utility(x, 1)
        expected_utility_no_loan = self.expected_utility(x, 0)

        if expected_utility_give_loan > expected_utility_no_loan:
            return 1
        else:
            return 0
