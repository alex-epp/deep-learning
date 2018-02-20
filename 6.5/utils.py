import numpy.matlib as np

def generate_data(xmin, xmax, npoints, estddev, esize, f):
    '''Generates random (x, y) ordered pairs that approximately follow a given
    function.

    Arguments:
    xmin -- Minimum generated x value
    xmax -- Maximum generated x value
    npoints -- Number of points to return
    estddev -- Standard deviation of generated y-error
    esize -- Size of y-error
    f -- function to be approximated

    Returns:
    A tuple of numpy arrays (xs, ys) where xs contains npoints values in the
    range [xmin, xmax), and ys contains the corresponding f(x) values, modified
    by random error of standard deviation estddev and size esize.
    '''
    xs = np.random.uniform(xmin, xmax, npoints)
    err = np.random.normal(0, estddev, npoints)*esize
    ys = [f(x) + e for x, e in zip(xs, err)]
    return np.array(xs), np.array(ys)