import MLP
import utils

import numpy.matlib as np

def classid(name):
    if name == 'Iris-setosa':
        return 0
    elif name == 'Iris-versicolor':
        return 1
    elif name == 'Iris-virginica':
        return 2
    else:
        print('Unrecognized class: ', name)

# Read training data
xs = []
ys = []
for line in open('data/iris.data'):
    toks = line.strip().split(',')
    xs.append([float(x) for x in toks[:4]])
    ys.append(classid(toks[4]))

xs = np.matrix(xs).reshape((-1, 4))
ys = np.matrix(ys).reshape((-1, 1))

# Shuffle data
p = np.random.permutation(np.arange(xs.shape[0]))
xs = xs[p]
ys = ys[p]

# Split into test and training sets
training_size = int(xs.shape[0]*100/80)
xs_train = xs[0:training_size]
ys_train = ys[0:training_size]
xs_test = xs[training_size:]
ys_test = ys[training_size:]

# Construct model
mlp = MLP.MLP()
w = 4
mlp.add_op(MLP.MatMul(utils.rmat(w, 4)))
mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
mlp.add_op(MLP.Rectify(w))
mlp.add_op(MLP.MatMul(np.rand(1, w)))
mlp.check_linkage((4,1))

# Train
stop = MLP.EarlyStopping(10, 20, 1000, xs_test, ys_test)
grad_descent =  MLP.Adam()
trainer = MLP.MLPTrainer(1, stop, grad_descent)
trainer.backprop(xs_train, ys_train, mlp)

# Check classification error
err = 0
for i in range(xs.shape[0]):
    x = xs[i].T
    y = ys[i].T

    if y != round(mlp.eval(x)[0,0]):
        err += 1

print('\nError rate: {}%'.format(err/xs.shape[0]*100))
