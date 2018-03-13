import MLP
import MLP.mnist
import utils

import numpy.matlib as np

def main():
    # Load model
    print('Loading model')
    mlp = MLP.MLP()
    mlp.load(open('saves/mlp1', 'rb'))

    # Read data
    print('Reading data')
    _, _, _, _, te_i, _ = MLP.mnist.load()

    
    while True:
        ex = MLP.mnist.from_BMP('custom_data/digit.bmp')
        #MLP.mnist.visualize(ex)
        probabilities = mlp.eval(ex.T)
        prediction = np.argmax(probabilities)
        probability = probabilities[prediction,0]*100
        print('{} ({:.2f}%)'.format(prediction, probability))
        input()
    '''
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
    '''

if __name__ == "__main__":
    main()