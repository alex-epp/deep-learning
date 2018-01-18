import numpy as np
import matplotlib.pyplot as plt
import random

class Polyfit:
    def __init__(self, degree, lambda_):
        self.degree = degree
        self.lambda_ = lambda_

    def train(self, xs, ys):
        X = np.matrix([np.power(x, range(self.degree+1)) for x in xs])
        y = np.matrix(ys).transpose()
        self.w = (X.T*X + np.eye(X.shape[1])*self.lambda_).I*X.T*y
    
    def predict(self, x):
        return np.sum(np.power(x, range(self.degree+1))*self.w)

    def error(self, xs, ys):
        return np.mean([(y-self.predict(x))**2 for x, y in zip(xs, ys)])

    def describe(self):
        return '{}-degree polyfit'.format(self.degree)

def generate_data(xmin, xmax, npoints, estddev, esize, f):
    xs = [random.uniform(xmin, xmax) for _ in range(npoints)]
    err = np.random.normal(0, estddev, npoints)*esize
    ys = [f(x) + e for x, e in zip(xs, err)]
    return np.array(xs), np.array(ys)

# Randomly generate data
ndata = 100
xs, ys = generate_data(0, 5, ndata, 1, 1, lambda x: x**2)

# Split into training/test data
p = np.random.permutation(xs.shape[0])
xs = xs[p]
ys = ys[p]
xs_train = xs[:int(ndata*0.8)]
ys_train = ys[:int(ndata*0.8)]
xs_test = xs[int(ndata*0.8):]
ys_test = ys[int(ndata*0.8):]

# Learn data function
learners = [Polyfit(i, 0) for i in range(0,6)]
for l in learners:
    l.train(xs_train, ys_train)
errors_test = [l.error(xs_test, ys_test) for l in learners]
errors_train = [l.error(xs_train, ys_train) for l in learners]

#Print errors calculated from test data
for l, e in zip(learners, errors_test):
    print('Error ({}): {}'.format(l.describe(), e))

# Plot x and y values from training data, as well as learned function
plt.subplot(211)
plt.plot(xs_train, ys_train, 'bo')
plt.plot(xs_test, ys_test, 'ro')
for l in learners:
    plt.plot(np.arange(0,6,0.1),
             [l.predict(x) for x in np.arange(0,6,0.1)])

# Plot errors
plt.subplot(212)
plt.plot(range(len(errors_train)), errors_train, 'bo')
plt.plot(range(len(errors_test)), errors_test, 'ro')

plt.show()