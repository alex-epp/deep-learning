import numpy.matlib as np
import os
from flask import Flask, send_from_directory, request

import MLP


filenames = ['saves/mlp{}'.format(i+1) for i in range(5)]
ensemble = MLP.Ensemble()
ensemble.load(open(n, 'rb') for n in filenames)

app = Flask(__name__, static_folder='../build')

@app.route('/predict', methods=['POST'])
def predict():
    response = np.matrix(list(float(item) for item in request.get_json().values()))
    probs = ensemble.eval(response.T)
    pred = np.argmax(probs)
    return '{} ({:.0f}% certainty)'.format(pred, probs[pred, 0]*100)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path == "":
        return send_from_directory('../build', 'index.html')
    else:
        if(os.path.exists("../build/" + path)):
            return send_from_directory('../build', path)
        else:
            return send_from_directory('../build', 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', use_reloader=True, port=5000, threaded=True)