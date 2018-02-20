import numpy as np


def grad_descent(start, f_grad, learning_rate, steps):
    '''
    Searches for a local minimum of some function
        f: R^n -> R
    using gradient descent.

    Keyword arguments:
        start         -- the value to start the search at. Must be given
                         as an array of size n.
        f_grad        -- a function specifying the gradient of f. Must
                         take and return a numpy array of size n.
        learning_rate -- the (fixed) learning rate.
        steps         -- the (fixed) number of gradient descent steps to
                         excecute.
    
    Usage: (this example finds the minimum of x^2 + y^2)
    >>> grad_f = lambda xy: np.array([2*xy[0], 2*xy[1]], dtype=np.float)
    >>> min_xy = grad_descent([1, 1], grad_f, 0.1, 100)
    >>> np.allclose(min_xy, 0)
    True
    '''
    x = np.array(start, dtype=np.float) # Copy input to avoid overwriting
    for _ in range(steps):
        x -= f_grad(x) * learning_rate

    return x


def approx_grad(f, epsilon):
    '''
    Generates a function that is a second-order numeric approximation of
    the gradient of the given function.

    Keyword arguments:
        f -- the given function f: R^n -> R. Must take a numpy array as
             input and return a single number.
        epsilon -- tuning parameter e, used in the directional
                   derivative approximation equation:

                    df/dr ~= (f(x+r*e) - f(x-r*e))/2e
        
    Usage:
    >>> f = lambda xy: xy[0]**2 + xy[1]**2
    >>> f_grad = lambda xy: [2*xy[0], 2*xy[1]]
    >>> f_grad_numeric = approx_grad(f, 0.00001)
    >>> z1, z2 = [], []
    >>> for xy in np.random.rand(100, 2):
    ...     z1.append(f_grad(xy))
    ...     z2.append(f_grad_numeric(xy))
    >>> np.allclose(z1, z2)
    True
    '''
    # Construct gradient function. This is the final function that will be
    # returned.
    def f_grad(x):
        # Construct directional derivative function for this particular x.
        def f_dir(i):
            ei = np.zeros_like(x)
            ei[i] = 1
            return (f(x+ei*epsilon) - f(x-ei*epsilon)) / (2*epsilon)
        # End f_dir

        # Vectorize directional derivative function, for easy conversion to
        # a gradient
        v_f_dir = np.vectorize(f_dir)
        
        # Return gradient constructed by applying directional derivatives to
        # each axis
        return v_f_dir(np.arange(x.shape[0]))
    # End f_grad

    # Now that gradient function is constructed, return it.
    return f_grad


def approx_grad_descent(start, f, learning_rate, steps, epsilon=0.001):
    '''
    As grad_descent, but takes the actual function f instead of its
    gradient. The gradient is approximated using approx_grad.

    Usage: (this example finds the minimum of x^2 + y^2)
    >>> grad_f = lambda xy: np.sum(xy**2)
    >>> min_xy = approx_grad_descent([1, 1], grad_f, 0.1, 100, 0.001)
    >>> np.allclose(min_xy, 0)
    True
    '''
    return grad_descent(start,
                        approx_grad(f, epsilon),
                        learning_rate,
                        steps)


if __name__ == "__main__":
    import doctest
    fc, _ = doctest.testmod()
    if fc is 0:
        print("grad_descent.py: all tests passed.")