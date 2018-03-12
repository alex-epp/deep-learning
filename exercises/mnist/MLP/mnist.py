import gzip
import numpy.matlib as np
import pandas
import pickle
import PIL

def load():
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


def visualize_batch(img_arrays):
    total_img = PIL.Image.new('L', (28*img_arrays.shape[0], 28))
    for i in range(img_arrays.shape[0]):
        img_array = np.reshape(np.array(img_arrays[i]*255, dtype=np.uint8), (28, 28))
        img = PIL.Image.fromarray(img_array, 'L')
        total_img.paste(img, (28*i, 0))
    
    total_img.show()

def visualize(img_array):
    img_array = np.reshape(np.array(img_array*255, dtype=np.uint8), (28, 28))
    img = PIL.Image.fromarray(img_array, 'L')
    img.show()

def reformat(img):
    # Find bounding box of interest
    

    return img

def from_BMP(filename):
    img = PIL.Image.open(filename)
    img = reformat(img)
    img_array = np.matrix(img.getdata()).reshape(img.size[0] * img.size[1], -1)
    return img_array[:,0].T / 255