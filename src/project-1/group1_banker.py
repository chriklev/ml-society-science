import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class Group1Banker:

    def __init__(self):
        """A simple constructor that initializes the decision maker class with-
        out the utility epsilon.
        """
        self._utility_epsilon_enabled = False

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

        self.model = self._fit_model(X, y)

        if self._utility_epsilon_enabled:
            self._utility_epsilon = self._calculate_utility_epsilon(
                max_alpha=self._max_type1_error)
        else:
            self._utility_epsilon = 0

    def _fit_model(self, X, y):
        """Fits the model

        Args:
            X: the covariates
            y: the response variable

        Notes:
            Using logistic regression, adapted from 
            https://scikit-learn.org/stable/modules/generated/sklearn.
            linear_model.LogisticRegression.html
        """
        log_reg_object = LogisticRegression(random_state=1, max_iter=2000)
        return log_reg_object.fit(X, y)

    def enable_utility_epsilon(self, max_alpha=0.05):
        """Enables the utility epsilon in in order to reduce the probability of
        type 1 error.

        Args:
            max_alpha: the maximum 'allowed' probability for type 1 errors.
        """
        self._utility_epsilon_enabled = True
        self._max_type1_error = max_alpha

    def _calculate_utility_epsilon(self, max_alpha=0.05):
        """Estimates the threshold to use in the utility calculations based on
        the training data. The method does this by splitting the training data
        into a training set and a validation set. The validation set is used in
        order to estimate the tuning parameter 'epsilon' which implicitly 
        calculates the estimated probability of type 1 error 'alpha_value' that
        should be below the threshold of 'max_alpha'.

        Args:
            max_alpha: the maximal probability for type 1 error that is allowed

        Returns:
            The estimated utility epsilon.
        """
        X = self.data[0]
        y = self.data[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.25)
        temp_model = self._fit_model(X_train, y_train)

        MAX_ITER = 100000
        COUNT_ITER = 0
        epsilon = 0
        delta_epsilon = 100

        # initial estimated alpha value
        alpha_value = 1

        while (alpha_value >= max_alpha) and COUNT_ITER < MAX_ITER:
            alpha_value = self._calculate_false_positive_rate(
                temp_model, X_test, y_test, epsilon)
            epsilon += delta_epsilon
            COUNT_ITER += 1

        return epsilon

    def _calculate_false_positive_rate(self, temp_model, X_test, y_test, eps):
        """Calculates the percentage of false positives among the results on 
        the test set from the training data.

        Args:
            temp_model: the model fitted with the training part of the training
            data
            X_test: the covariates in the test part of the training data
            y_test: the test part in the test part of the training data
            eps: the epsilon (threshold) to use when deciding the best action
            of the policy
        """
        false_positives = 0
        for i in range(len(X_test)):
            test_action = self._calculate_action(
                temp_model, X_test.iloc[i], eps)

            if y_test.iloc[i] == 0 and test_action == 1:
                false_positives += 1

        return false_positives/len(y_test)

    def _calculate_action(self, model, x_test, eps=0):
        """Calculates the best action based on a specific epsilon (threshold).

        Args:
            model: the model to use when predicting the best action
            x_test: the test set to use when deciding the best action
            eps: the threshold to use when deciding the best action

        Returns:
            The best action dependent on the epsilon threshold.
        """
        p_c = model.predict_proba(self._reshape(x_test))[0][1]

        r = self.rate
        # duration in months
        n = x_test['duration']
        # amount
        m = x_test['amount']

        e_x = p_c * m * ((1 + r) ** n - 1) + (1 - p_c) * (-m)

        if e_x > eps:
            return 1
        else:
            return 0

    # set the interest rate
    def set_interest_rate(self, rate):
        """Sets the interest rate for the decision maker.

        Args:
            rate: the interest rate to use in the calculations.
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
        """Calculate expected utility using the decision maker model.

        Args:
            x: A new observation.
            action: Whether or not to grant the loan.

        Returns:
            The expected utility of the decision maker.
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
        An epsilon for utility is also set as the threshold that the expected
        utility should exceed in order to get the best action. This utility
        epsilon is 0 if the banker is not configured to use this functionality.
        Otherwise it is estimated from the training data as the value that 
        provide a type 1 error below the parameter '_max_type1_error'.

        Args:
            x: A new observation.
        Returns:
            Best action based on maximizing utility.
        """
        expected_utility_give_loan = self.expected_utility(x, 1)
        expected_utility_no_loan = self.expected_utility(x, 0)

        if expected_utility_give_loan > (expected_utility_no_loan + self._utility_epsilon):
            return 1
        else:
            return 0
