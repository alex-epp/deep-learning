import MLP
import numpy.matlib as np
import progressbar


def rmat(n, m):
    m = 2*np.rand(n, m) - np.ones((n, m))
    #print(m)
    return m


# Construct model
mlp = MLP.MLP()
w = 5

mlp.add_op(MLP.MatMul(rmat(w, 2)))
mlp.add_op(MLP.VecAdd(rmat(w, 1)))
mlp.add_op(MLP.Rectify(w))

mlp.add_op(MLP.MatMul(np.rand(1, w)))

mlp.check_linkage((2,1))

# Input data with 4 examples
X = np.matrix([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])

# Correct output
Y = np.matrix([[0],
               [1],
               [1],
               [0]])

# Train
bar = progressbar.ProgressBar()
for _ in bar(range(1000)):
    mlp.backprop(X, Y, 0.01)

# Print examples with solutions
for x in X:
    x = np.matrix(x).T
    print("{} XOR {} = {:.0f}".format(x[0,0], x[1,0], mlp.eval(x)[0,0]))