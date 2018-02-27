import MLP
import utils

import matplotlib.pyplot as plt
import numpy.matlib as np


# Generate training data
ndata = 100
xs, ys = utils.generate_data(-10, 10, ndata, 0, 0, lambda x: np.sin(x))

training_size = 80
xs_train = xs[0:training_size]
ys_train = ys[0:training_size]
xs_test = xs[training_size:]
ys_test = ys[training_size:]


# Construct model
mlp = MLP.MLP()
w = 100
mlp.add_op(MLP.MatMul(utils.rmat(w, 1)))
mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
mlp.add_op(MLP.Rectify(w))
mlp.add_op(MLP.MatMul(np.rand(1, w)))
mlp.check_linkage((1, 1))

# Train
stop = MLP.EarlyStopping(10, 200, 10000, xs_test, ys_test)
grad_descent = MLP.Adam()
trainer = MLP.MLPTrainer(1, stop, grad_descent)
trainer.backprop(xs_train, ys_train, mlp)

# Plot
plt.plot(xs, ys, 'bo')
xs = np.linspace(-10, 10, 10000)
plt.plot(xs, [mlp.eval(x)[0, 0] for x in xs], 'r')
plt.show()
