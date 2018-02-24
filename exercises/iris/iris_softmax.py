import MLP
import utils

import numpy.matlib as np

def classid(name):
    if name == 'Iris-setosa':
        return [1, 0, 0]
    elif name == 'Iris-versicolor':
        return [0, 1, 0]
    elif name == 'Iris-virginica':
        return [0, 0, 1]
    else:
        print('Unrecognized class: ', name)

def main(debug=False):
    # Read training data
    xs = []
    ys = []
    for line in open('data/iris.data'):
        toks = line.strip().split(',')
        xs.append([float(x) for x in toks[:4]])
        ys.append(classid(toks[4]))

    xs = np.matrix(xs, dtype=float).reshape((-1, 4))
    ys = np.matrix(ys, dtype=float).reshape((-1, 3))

    # Shuffle data
    p = np.random.permutation(np.arange(xs.shape[0]))
    xs = xs[p]
    ys = ys[p]

    # Split into test and training sets
    training_size = int(xs.shape[0]*80/100)
    xs_train = xs[0:training_size]
    ys_train = ys[0:training_size]
    xs_test = xs[training_size:]
    ys_test = ys[training_size:]

    # Construct model
    mlp = MLP.MLP()
    w = 6
    d = 1
    mlp.add_op(MLP.MatMul(utils.rmat(w, 4)))
    mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
    mlp.add_op(MLP.Rectify(w))

    for _ in range(d-1):
        mlp.add_op(MLP.MatMul(utils.rmat(w, w)))
        mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
        mlp.add_op(MLP.Rectify(w))

    mlp.add_op(MLP.MatMul(np.rand(3, w)))
    mlp.add_op(MLP.SoftMax(3))

    # Train
    stop = MLP.EarlyStopping(10, 10, 1000, xs_test, ys_test)
    grad_descent =  MLP.Adam()
    cost = MLP.SoftMaxCrossEntropy()
    trainer = MLP.MLPTrainer(1, stop, grad_descent, cost)
    trainer.backprop(xs_train, ys_train, mlp)

    # Check classification error
    err = 0
    for i in range(xs.shape[0]):
        x = xs[i].T
        y = ys[i].T

        if (y != np.around(mlp.eval(x))).any():
            err += 1
            if debug:
                print(y.T)
                print(np.around(mlp.eval(x).T, 2))
        
    print('\nError rate: {}%'.format(err/xs.shape[0]*100))

if __name__ == "__main__":
    main(debug=False)