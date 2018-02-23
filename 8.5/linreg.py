import MLP
import utils

import matplotlib.pyplot as plt
import numpy.matlib as np
import progressbar


# Generate training data
ndata = 100
xs, ys = utils.generate_data(0, 5, ndata, 1, 1, lambda x: 2*x + 4)

# Construct model
mlp = MLP.MLP()
mlp.add_op(MLP.MatMul(np.rand(1, 1)))
mlp.add_op(MLP.VecAdd(np.rand(1, 1)))

# Train
bar = progressbar.ProgressBar()
grad_descent = MLP.AdaGrad(1)
for _ in bar(range(1)):
    mlp.backprop(xs, ys, grad_descent)

print("Learned slope: {}".format(mlp.operations[0].value[0,0]))
print("Learned intercept: {}".format(mlp.operations[1].value[0,0]))

plt.plot(xs, ys, 'bo')
plt.plot([0, 5], [4, 14], 'b')
plt.plot(sorted(xs), [mlp.eval(x)[0,0] for x in sorted(xs)], 'r')
plt.show()