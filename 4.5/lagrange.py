import numpy as np
from grad_descent import approx_grad, approx_grad_descent


def lagrange_descent(start_x, start_l, f, g, learning_rate, steps,
                     epsilon=0.001):
    '''
    Finds a local minimum of f(x), constrained to g(x) = 0. This is done
    by gradient descent on the function:

                h: (R^n, l) -> R, (x, l) -> || L(x, l) ||^2
    
    where L is the Lagrangian:

                L: (R^n, l) -> R, (x, l) -> f(x) + l*g(x)
    
    Note: this is pretty inaccurate, probably because of the repeated
          gradient approximations and naive gradient descent backend.
    
    Keyword arguments:
        start_x       -- Initial x value the gradient descent begins at.
                         Must be a numpy array with n elements.
        start_l       -- Initial l value the gradient descent begins at.
        f             -- The function f. Must take a numpy array with n
                         elements as input and return a single number.
        g             -- The function g. Must take a numpy array with n
                         elements as input and return a single number.
        learning_rate -- Gradient descent learning rate.
        steps         -- Gradient descent steps.
        epsilon       -- Gradient estimation parameter.
    
    Returns: min_x, min_l
        min_x -- x-value at the local minimum
        min_l -- l-value at the local minimum

    Usage: (finds a minimum of f(x, y) = x^2+y^2 such that x^2+y^2=1)
    >>> f = lambda x: np.sum(x**2)
    >>> g = lambda x: np.sum(x**2) - 1
    >>> x, _ = lagrange_descent(np.array([10.0, 10.0]), 0.0,
    ...                         f, g,
    ...                         0.001, 300, 0.00001)
    >>> np.isclose(g(x), 0, atol=0.1)
    True
    >>> np.isclose(f(x), 1, atol=0.1)
    True
    '''
    # Lagrange function L(x, l)
    def lagrange(xl):
        x = xl[:-1]
        l = xl[-1]
        return f(x) + l*g(x)

    # h(x, l)
    def h(xl):
        lagrange_grad = approx_grad(lagrange, epsilon)
        return np.sum(lagrange_grad(xl)**2)
    
    # Do gradient descent
    start_xl = np.append(start_x, [start_l])
    min_xl = approx_grad_descent(start_xl, h, learning_rate, steps)
    min_x = min_xl[:-1]
    min_l = min_xl[-1]

    # Return values at local minimum
    return min_x, min_l


def main():
    def f(x):
        return np.sum(x**2)
    
    def g(x):
        return np.sum((x-2)**2) - 1
    
    start_x = np.array([0.0])
    start_l = 1
    
    min_x, _ = lagrange_descent(start_x, start_l, f, g, 0.001, 10000, 0.0001)

    print('Minimum: f({}) = {}'.format(min_x, f(min_x)))


if __name__ == '__main__':
    import doctest
    fc, _ = doctest.testmod()
    if fc is 0:
        print("lagrange.py: all tests passed.")
        main()
