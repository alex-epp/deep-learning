import MLP
import MLP.mnist
import utils

import numpy.matlib as np
import sys


def main():
    model_name = 'mlp3' if len(sys.argv) <= 1 else sys.argv[1]

    # Read data
    print('Reading data')
    tr_i, tr_o, va_i, va_o, te_i, te_o = MLP.mnist.load(augment=False)
    
    # Construct model
    print('Constructing model')
    mlp = MLP.MLP()
    w = 784
    mlp.add_op(MLP.MatMul(utils.rmat(w, 784)))
    mlp.add_op(MLP.VecAdd(utils.rmat(w, 1)))
    mlp.add_op(MLP.Rectify(w))

    mlp.add_op(MLP.MatMul(np.rand(10, w)))
    mlp.add_op(MLP.SoftMax(10))

    # Train
    print('Training model')
    # stop = MLP.EarlyStopping(1, 15, 100, va_i, va_o)
    stop = MLP.FixedStopping(10)
    grad_descent = MLP.Adam()
    cost = MLP.SoftMaxCrossEntropy()
    visual = MLP.BarVisualizer()
    reg = None
    # reg = MLP.L2Regularization(.5/1000)
    # visual = MLP.GraphVisualizer(tr_i, tr_o, va_i, va_o, cost)
    trainer = MLP.MLPTrainer(200, stop, grad_descent, cost, visual, reg)
    trainer.backprop(tr_i, tr_o, mlp)

    # Check classification error
    print('Calculating classifying error')
    err_te, _ = utils.classifier_error(mlp, te_i, te_o)
    print('\nError rate: (test) {}%'.format(err_te*100))

    err_va, _ = utils.classifier_error(mlp, va_i, va_o)
    print('\nError rate (validation): {}%'.format(err_va*100))

    err_tr, _ = utils.classifier_error(mlp, tr_i, tr_o)
    print('\nError rate (train): {}%'.format(err_tr*100))

    # Save classifier
    print('Saving model')
    mlp.save(open('saves/'+model_name, 'wb'))

if __name__ == "__main__":
    main()