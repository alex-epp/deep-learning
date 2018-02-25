import MLP
import utils

import gzip
import numpy.matlib as np
import pandas
import pickle


def load_mnist():
    '''
    Loads the mnist datasets, as in
    http://www.deeplearning.net/tutorial/gettingstarted.html
    '''
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()

    train_inputs = np.matrix(train_set[0], dtype=np.float32)
    train_outputs = np.matrix(pandas.get_dummies(train_set[1]).values, dtype=np.float32)
    
    valid_inputs = np.matrix(valid_set[0], dtype=np.float32)
    valid_outputs = np.matrix(pandas.get_dummies(valid_set[1]).values, dtype=np.float32)

    test_inputs = np.matrix(test_set[0], dtype=np.float32)
    test_outputs = np.matrix(pandas.get_dummies(test_set[1]).values, dtype=np.float32)

    return (train_inputs, train_outputs,
            valid_inputs, valid_outputs,
            test_inputs, test_outputs)

def main():
    # Read data
    print('Reading data')
    tr_i, tr_o, va_i, va_o, te_i, te_o = load_mnist()
    
    # Construct model
    print('Constructing model')
    mlp = MLP.MLP()
    w = 16
    d = 4
    mlp.add_op(MLP.MatMul(utils.rmat(w, 784)))
    mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
    mlp.add_op(MLP.Rectify(w))

    for _ in range(d-1):
        mlp.add_op(MLP.MatMul(utils.rmat(w, w)))
        mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
        mlp.add_op(MLP.Rectify(w))

    mlp.add_op(MLP.MatMul(np.rand(10, w)))
    mlp.add_op(MLP.SoftMax(10))

    # Train
    print('Training model')
    stop = MLP.FixedStopping(30)
    grad_descent =  MLP.Adam()
    cost = MLP.SoftMaxCrossEntropy()
    trainer = MLP.MLPTrainer(100, stop, grad_descent, cost)
    trainer.backprop(tr_i, tr_o, mlp)

    # Check classification error
    print('Calculating classifying error')
    err, _ = utils.classifier_error(mlp, te_i, te_o)
    print('\nError rate: {}%'.format(err/te_i.shape[0]*100))


if __name__ == "__main__":
    main()