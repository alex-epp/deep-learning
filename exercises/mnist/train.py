import MLP
import MLP.mnist
import utils

import numpy.matlib as np


def main():
    # Read data
    print('Reading data')
    tr_i, tr_o, va_i, va_o, te_i, te_o = MLP.mnist.load()
    
    # Construct model
    print('Constructing model')
    mlp = MLP.MLP()
    w = 32
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
    stop = MLP.EarlyStopping(1, 10, 100, va_i, va_o)
    #stop = MLP.FixedStopping(100)
    grad_descent =  MLP.Adam()
    cost = MLP.SoftMaxCrossEntropy()
    visual = MLP.BarVisualizer()
    reg = MLP.L2Regularization(.5/60000)
    #visual = MLP.GraphVisualizer(tr_i, tr_o, va_i, va_o, cost)
    trainer = MLP.MLPTrainer(100, stop, grad_descent, cost, visual, reg)
    trainer.backprop(tr_i, tr_o, mlp)

    # Check classification error
    print('Calculating classifying error')
    err_te, _ = utils.classifier_error(mlp, te_i, te_o)
    print('\nError rate: (test) {}%'.format(err_te/te_i.shape[0]*100))

    err_va, _ = utils.classifier_error(mlp, va_i, va_o)
    print('\nError rate (validation): {}%'.format(err_va/va_i.shape[0]*100))

    err_tr, _ = utils.classifier_error(mlp, tr_i, tr_o)
    print('\nError rate (train): {}%'.format(err_tr/tr_i.shape[0]*100))

    # Save classifier
    print('Saving model')
    mlp.save(open('saves/mlp', 'wb'))

if __name__ == "__main__":
    main()