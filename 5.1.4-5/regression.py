import numpy as np
import matplotlib.pyplot as plt

import polyfit


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


def main():
    # Randomly generate data
    ndata = 100
    xs, ys = generate_data(0, 5, ndata, 3, 3, lambda x: x**2)

    # Split into training/test data
    # p = np.random.permutation(xs.shape[0])
    # xs = xs[p]
    # ys = ys[p]
    xs_train = xs[:int(ndata*0.8)]
    ys_train = ys[:int(ndata*0.8)]
    xs_test = xs[int(ndata*0.8):]
    ys_test = ys[int(ndata*0.8):]

    # Learn data function
    learners = [polyfit.Polyfit(i, 0) for i in range(0, 4)]
    for l in learners:
        l.train(xs_train, ys_train)
    errors_test = [l.error(xs_test, ys_test) for l in learners]
    errors_train = [l.error(xs_train, ys_train) for l in learners]

    # Print errors calculated from test data
    for l, e_train, e_test in zip(learners, errors_train, errors_test):
        print('Error ({}): {}\t{}'.format(l.describe(), e_train, e_test))

    # Plot x and y values from test and training data, as well as learned function
    plt.subplot(211)
    plt.plot(xs_train, ys_train, 'bo')
    plt.plot(xs_test, ys_test, 'ro')
    for l in learners:
        plt.plot(np.arange(0, 6, 0.1),
                 [l.predict(x) for x in np.arange(0, 6, 0.1)])

    # Plot errors
    plt.subplot(212)
    plt.plot(errors_train, 'bo')
    plt.plot(errors_test, 'ro')

    plt.show()


if __name__ == "__main__":
    main()
