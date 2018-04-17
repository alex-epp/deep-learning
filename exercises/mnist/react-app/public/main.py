from flask import Flask, send_from_directory, request
import numpy.matlib as np

import sys
sys.path.append('backend/')

import MLP
# import MLP.mnist

app = Flask(__name__)

filenames = ['backend/saves/mlp{}'.format(i+1) for i in range(5)]
ensemble = MLP.Ensemble()
ensemble.load(open(n, 'rb') for n in filenames)


@app.route('/predict', methods=['POST'])
def predict():
    img_dict = request.get_json()
    response = np.matrix(list(float(img_dict[key]) for key in sorted(img_dict)))
    return str(response)
    # MLP.mnist.visualize(response)
    probs = ensemble.eval(response.T)
    pred = np.argmax(probs)
    return '{} ({:.0f}% certainty)'.format(pred, probs[pred, 0]*100)


@app.route('/')
def serve():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run()
    # app.run(host='0.0.0.0', use_reloader=True, port=5000, threaded=True)
