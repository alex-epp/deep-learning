import numpy as np
import matplotlib.pyplot as plt
import random

# Randomly generate training data
xs = np.array([1, 2, 3, 4, 5])
ys = np.array([v+random.uniform(-1, 1)+10 for v in [2, 4, 6, 8, 10]])

# Create matrix of inputs X_train and matrix of desired outputs
# y_train.
X_train = np.matrix([[x, 1] for x in xs])
y_train = np.matrix(ys).transpose()

# For some x in X_train, y in y_train, the predicted y, y^, is
# calculated from a matrix if weights w:
#   y^ = w.transpose() * x
# Set w to minimize the squared error:
w = (X_train.transpose()*X_train).getI()*X_train.transpose()*y_train
w1, w2 = w[0, 0], w[1, 0]

# Plot x and y values from training data, as well as linear regression
# line.
plt.plot(xs, ys, 'ro')
plt.plot(np.array([1, 5]), np.array([1, 5])*w1+w2)
plt.plot()
plt.show()