import numpy as np

# Model values manually set to the global minimum of the MSE
W = np.matrix([[1, 1], [1, 1]])
c = np.matrix([[0], [-1]])
w = np.matrix([[1], [-2]])
b = 0

# Input data with 4 examples
X = np.matrix([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])

# Calculate output for each input
y = np.apply_along_axis(lambda x: w.T*np.maximum(0, W.T*x.T+c)+b,
                          axis=1,
                          arr=X,
                          )

# Print examples with solutions
for i in range(4):
    print("{} XOR {} = {}".format(X[i,0], X[i,1], y[i,0]))