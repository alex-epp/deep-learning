import MLP
import MLP.mnist

import numpy.matlib as np


def main():
    # Load model
    print('Loading model')
    mlp = MLP.MLP()
    mlp.load(open('saves/mlp1', 'rb'))

    # Read data
    print('Reading data')
    _, _, _, _, te_i, _ = MLP.mnist.load()

    # Visualize elements
    while True:
        print('Predicting')
        ex = np.random.permutation(te_i.shape[0])[:10]
        MLP.mnist.visualize_batch(te_i[ex])
        predictions = []
        for i in range(ex.shape[0]):
            probabilities = mlp.eval(te_i[ex[i]].T)
            predictions.append(np.argmax(probabilities))
        print(*predictions)
        input()


if __name__ == "__main__":
    main()
