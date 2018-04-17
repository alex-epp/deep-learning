from flask import Flask, send_from_directory, request
import numpy.matlib as np

import os
os.pathname.append('../MLP/')
from MLP.ensemble import Ensemble


app = Flask(__name__)

filenames = ['../MLP/saves/mlp{}'.format(i+1) for i in range(5)]
ensemble = Ensemble()
ensemble.load(open(n, 'rb') for n in filenames)


@app.route('/predict', methods=['POST'])
def predict():
    response = np.matrix(list(float(item) for item in request.get_json().values()))
    probs = ensemble.eval(response.T)
    pred = np.argmax(probs)
    return '{} ({:.0f}% certainty)'.format(pred, probs[pred, 0]*100)


@app.route('/')
def serve():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run()
    # app.run(host='0.0.0.0', use_reloader=True, port=5000, threaded=True)
