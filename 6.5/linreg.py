import MLP

import matplotlib.pyplot as plt
import numpy.matlib as np
import progressbar


def generate_data(xmin, xmax, npoints, estddev, esize, f):
    '''Generates random (x, y) ordered pairs that approximately follow a given
    function.

    Arguments:
    xmin -- Minimum generated x value
    xmax -- Maximum generated x value
    npoints -- Number of points to return
    estddev -- Standard deviation of generated y-error
    esize -- Size of y-error
    f -- function to be approximated

    Returns:
    A tuple of numpy arrays (xs, ys) where xs contains npoints values in the
    range [xmin, xmax), and ys contains the corresponding f(x) values, modified
    by random error of standard deviation estddev and size esize.
    '''
    xs = np.random.uniform(xmin, xmax, npoints)
    err = np.random.normal(0, estddev, npoints)*esize
    ys = [f(x) + e for x, e in zip(xs, err)]
    return np.array(xs), np.array(ys)


# Generate training data
ndata = 100
xs, ys = generate_data(0, 5, ndata, 1, 1, lambda x: 2*x + 4)

# Construct model
mlp = MLP.MLP()
mlp.add_op(MLP.MatMul(np.rand(1, 1)))
mlp.add_op(MLP.VecAdd(np.rand(1, 1)))

# Train
bar = progressbar.ProgressBar()
for _ in bar(range(100)):
    mlp.backprop(xs, ys, 0.1)

print("Learned slope: {}".format(mlp.operations[0].weights[0,0]))
print("Learned intercept: {}".format(mlp.operations[1].bias[0,0]))

plt.plot(xs, ys, 'bo')
plt.plot([0, 5], [4, 14], 'b')
plt.plot(sorted(xs), [mlp.eval(x)[0,0] for x in sorted(xs)], 'r')
plt.show()