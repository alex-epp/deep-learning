import MLP
import numpy.matlib as np
import progressbar
import utils

# Construct model
mlp = MLP.MLP()
w = 5

mlp.add_op(MLP.MatMul(utils.rmat(w, 2)))
mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
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
trainer = MLP.MLPTrainer(1000, 1, MLP.Adam())
trainer.backprop(X, Y, mlp)

# Print examples with solutions
for x in X:
    x = np.matrix(x).T
    print("{} XOR {} = {:.0f}".format(x[0,0], x[1,0], mlp.eval(x)[0,0]))