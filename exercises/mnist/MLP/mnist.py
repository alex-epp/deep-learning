import gzip
import numpy.matlib as np
import pandas
import pickle
import PIL
import skimage.transform
import scipy.ndimage.measurements

def load(augment=False):
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

    if augment:
        train_inputs, train_outputs = apply_augment(
            train_inputs, train_outputs
        )

    return (train_inputs, train_outputs,
            valid_inputs, valid_outputs,
            test_inputs, test_outputs)

def apply_augment(inputs, outputs):
    new_inputs = np.matrix(inputs, copy=True)
    new_outputs = np.matrix(outputs, copy=True)

    for i in range(new_inputs.shape[0]):
        angle = np.random.uniform(-45, 45)
        new_inputs[i] = np.reshape(
            skimage.transform.rotate(
                np.reshape(new_inputs[i], (28,-1)),
                angle
            ),
            784
        )
        #visualize_batch(np.array([inputs[i], new_inputs[i]]))
        #input()

    return (np.concatenate((inputs, new_inputs)),
            np.concatenate((outputs, new_outputs)))

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
    img = img.crop(img.getbbox())
    
    r = min(20/img.width, 20/img.height)
    img = img.resize(( int(img.width*r), int(img.height*r) ))

    m = np.sum(np.asarray(img), -1) / 255*3
    m = m / np.sum(np.sum(m))

    center = ( np.sum(np.sum(m, 0) * np.arange(img.width)),
               np.sum(np.sum(m, 1) * np.arange(img.height)) )

    center = int(center[0]), int(center[1])

    total_image = PIL.Image.new('L', (28,28), color=0)
    total_image.paste(
        img,
        box=(14 - center[0], 14 - center[1])
    )
    return total_image

def from_BMP(filename):
    img = PIL.Image.open(filename)
    img = reformat(img)
    img_array = np.matrix(img.getdata()).reshape(img.size[0] * img.size[1], -1)
    return img_array[:,0].T / 255