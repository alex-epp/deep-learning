import numpy as np


class Polyfit:
    def __init__(self, degree, lambda_):
        '''Initialize.

        Arguments:
        degree  -- Maximum degree of the polynomial
        lambda_ -- Weight decay parameter
        '''
        self.degree = degree
        self.lambda_ = lambda_
        self.w = [0 for _ in range(self.degree+1)]

    def train(self, xs, ys):
        '''Fits data to a polynomial given training data.

        Arguments:
        xs -- 1-dimensional array of x values
        ys -- 1-dimensional array of corresponding y values
        '''
        X = np.matrix([np.power(x, range(self.degree+1)) for x in xs])
        y = np.matrix(ys).transpose()
        self.w = (X.T*X + np.eye(X.shape[1])*self.lambda_).I*X.T*y

    def predict(self, x):
        '''Predicts new y values after train()ed.

        Arguments:
        x -- New x values

        Returns:
        Predicted y value at the given x value
        '''
        return np.sum(np.power(x, range(self.degree+1))*self.w)

    def error(self, xs, ys):
        '''Returns MSE of the model from the given test data.

        Arguments:
        xs -- 1-dimensional array of x values
        ys -- 1-dimensional array of y values

        Returns:
        Mean squared error of the model's predicted y-values compared to the
        actual y-values in the test dataset.
        '''
        return np.mean([(y-self.predict(x))**2 for x, y in zip(xs, ys)])

    def describe(self):
        '''Returns a string describing the model'''
        return '{}-degree polyfit'.format(self.degree)
