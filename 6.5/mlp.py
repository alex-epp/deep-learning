import numpy.matlib as np

class MatMul:
    '''
    >>> m = MatMul(np.matrix([[1.0, 0.0], [2.0, -1.0], [0.0, 3.0]]))
    >>> i = np.matrix([[1.0], [10.0]])
    >>> print(m.eval(i))
    [[  1.]
     [ -8.]
     [ 30.]]
    >>> np.array_equal(m.i, i)
    True
    >>> print(m.jacobian())
    [[ 1.  0.]
     [ 2. -1.]
     [ 0.  3.]]
    '''
    def __init__(self, weights):
        self.weights = np.matrix(weights)
        self.i = None

    def eval(self, i):
        self.i = i
        return self.weights*i
    
    def jacobian(self):
        return self.weights

    def shape(self):
        '''Returns (size of output, size of input)'''
        return self.weights.shape
    
    def grad_descent(self, J, learning_rate):
        grad = np.zeros(self.weights.shape)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                grad[i,j] = J[0,i] * self.i[j,0]
                
        self.weights = self.weights - grad * learning_rate

class VecAdd:
    '''
    >>> v = VecAdd(np.matrix([[0.0], [1.0], [2.0], [3.0]]))
    >>> i = np.matrix([[5.0], [7.0], [9.0], [11.0]])
    >>> print(v.eval(i))
    [[  5.]
     [  8.]
     [ 11.]
     [ 14.]]
    >>> np.array_equal(v.i, i)
    True
    >>> print(v.jacobian())
    [[ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0.  1.]]
    '''
    def __init__(self, bias):
        self.bias = np.matrix(bias)
        self.i = 0
    
    def eval(self, i):
        self.i = i
        return i + self.bias
    
    def jacobian(self):
        n = self.bias.shape[0]
        return np.eye(n)
    
    def shape(self):
        '''Returns (size of output, size of input)'''
        n = self.bias.shape[0]
        return (n, n)
    
    def grad_descent(self, J, learning_rate):
        self.bias = self.bias - (J*self.jacobian()).T * learning_rate

class Rectify:
    '''
    >>> r = Rectify(5)
    >>> i = np.matrix([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    >>> print(r.eval(i))
    [[ 0.]
     [ 0.]
     [ 0.]
     [ 1.]
     [ 2.]]
    >>> np.array_equal(r.i, i)
    True
    >>> print(r.jacobian())
    [[ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.]
     [ 0.  0.  0.  0.  1.]]
    '''
    def __init__(self, n):
        self.o = None
        self.i = None
        self.n = n
    
    def eval(self, i):
        self.i = i
        self.o = np.maximum(0, i)
        return self.o
    
    def jacobian(self):
        J = np.zeros((self.n, self.n))
        if self.o is not None:
            for i in range(self.n):
                J[i,i] = np.sign(self.o[i,0])
            
        return J
    
    def shape(self):
        '''Returns (size of output, size of input)'''
        return (self.n, self.n)
    
    def grad_descent(self, J, learning_rate):
        pass

class MLP:
    '''
    >>> def rmat(n, m):
    ...    m = 2*np.rand(n, m) - np.ones((n, m))
    ...    return m
    >>>
    >>> mlp = MLP()
    >>> mlp.add_op(MatMul(rmat(20, 2)))
    >>> mlp.add_op(VecAdd(rmat(20, 1)))
    >>> mlp.add_op(Rectify(20))
    >>> mlp.add_op(MatMul(rmat(1, 20)))
    >>> mlp.add_op(VecAdd(rmat(1, 1)))
    >>> x = np.matrix([[1], [0]])
    >>> y = mlp.eval(x)
    >>> y.shape
    (1, 1)
    >>> mlp.check_linkage((2, 1))
    >>>
    >>> mlp = MLP()
    >>> mlp.add_op(MatMul(rmat(20, 2)))
    >>> mlp.add_op(VecAdd(rmat(20, 1)))
    >>> mlp.add_op(Rectify(20))
    >>> mlp.add_op(MatMul(np.rand(1, 20)))
    >>> mlp.check_linkage((2,1))
    '''
    def __init__(self):
        self.operations = []
    
    def add_op(self, op):
        self.operations.append(op)
    
    def check_linkage(self, input_shape, output_shape=(1, 1)):
        '''
        WARNING: untested with multiple outputs.
        '''
        curr_shape = input_shape
        # Check forward propagation shape
        for op in self.operations:
            if op.shape()[1] is not curr_shape[0]:
                raise RuntimeError("invalid linkage: op.shape() = {}, curr_shape = {}" \
                                    .format(op.shape(), curr_shape))
            else:
                curr_shape = (op.shape()[0], curr_shape[1])
        
        # Final output must match
        if curr_shape != output_shape:
            raise RuntimeError("Invalid linkage: curr_shape = {}, output_shape = {}" \
                               .format(curr_shape, output_shape))
        
        curr_shape = output_shape[::-1]
        # Check back propagation shape
        for op in reversed(self.operations):
            if op.jacobian().shape[0] is not curr_shape[1]:
                raise RuntimeError("Invalid linkage: jacobian.shape = {}, curr_shape = {}" \
                                    .format(op.jacobian().shape, curr_shape))
            else:
                curr_shape = (curr_shape[0], op.jacobian().shape[1])
        
        # Input must match
        if curr_shape != input_shape[::-1]:
            raise RuntimeError("Invalid linkage: curr_shape = {}, input_shape = {}" \
                               .format(curr_shape, input_shape))

    def eval(self, X):
        out = X
        for op in self.operations:
            out = op.eval(out)
        
        return out
    
    def backprop(self, Xs, Ys, learning_rate = 0.1):
        for i in np.random.permutation(len(Xs)):
            X = np.matrix(Xs[i]).T
            Y = np.matrix(Ys[i]).T
            self.backprop_example(X, Y, learning_rate)
        
    def backprop_example(self, X, Y, learning_rate):
        J = self.eval(X) - Y
        for op in reversed(self.operations):
            op.grad_descent(J, learning_rate)
            J = J * op.jacobian()
            
if __name__ == "__main__":
    import doctest
    doctest.testmod()