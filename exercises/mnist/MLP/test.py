import MLP
import MLP.mnist
import utils

import numpy.matlib as np


def main():
    # Load model
    filenames = ['saves/mlp{}'.format(i+1) for i in range(5)]
    model = MLP.Ensemble()
    model.load(open(n, 'rb') for n in filenames)

    # Read data
    print('Reading data')
    _, _, _, _, te_i, te_o = MLP.mnist.load()

    # Calculate test error
    print('Calculating error')
    err_te, failures = utils.classifier_error(model, te_i, te_o)
    print('Error rate: {}% ({} failures)'.format(err_te*100, len(failures)))

    # Visualize elements
    while input() != 'exit':
        print('Predicting')
        ex = np.random.permutation(te_i.shape[0])[:10]
        MLP.mnist.visualize_batch(te_i[ex])
        predictions = []
        for i in range(ex.shape[0]):
            probabilities = model.eval(te_i[ex[i]].T)
            predictions.append(np.argmax(probabilities))
        print(*predictions)


if __name__ == "__main__":
    main()
