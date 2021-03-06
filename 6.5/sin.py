import MLP
import utils

import matplotlib.pyplot as plt
import numpy.matlib as np
import progressbar


# Generate training data
ndata = 100
xs, ys = utils.generate_data(-10, 10, ndata, 0, 0, lambda x: np.sin(x))

# Construct model
mlp = MLP.MLP()
w = 100
mlp.add_op(MLP.MatMul(utils.rmat(w, 1)))
mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
mlp.add_op(MLP.Rectify(w))
mlp.add_op(MLP.MatMul(np.rand(1, w)))
mlp.check_linkage((1,1))

# Train
bar = progressbar.ProgressBar()
for _ in bar(range(10000)):
    mlp.backprop(xs, ys, .001)

plt.plot(xs, ys, 'bo')
plt.plot(sorted(xs), [mlp.eval(x)[0,0] for x in sorted(xs)], 'r')
plt.show()