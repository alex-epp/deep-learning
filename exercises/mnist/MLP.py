import numpy.matlib as np
import progressbar

from numba import jit, jitclass

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
        self.value = np.matrix(weights, dtype=np.float32)
        self.i = None
        

    def eval(self, i):
        self.i = i
        return self.value*i
    
    def apply_jacobian(self, J):
        return J * self.value

    def shape(self):
        '''Returns (size of output, size of input)'''
        return self.value.shape
    
    def self_grad(self, J):
        '''
        Returns 'gradient' (may actually be arranged as a 2D matrix),
        i.e. the effect of each element of the weights matrix on the
        output.
        '''
        '''
        for i in range(self.grad.shape[0]):
            for j in range(self.grad.shape[1]):
                self.grad[i,j] = J[0,i] * self.i[j,0]
        '''
        return np.multiply(J, self.i).T

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
        self.value = np.matrix(bias, dtype=np.float32)
        self.i = 0
    
    def eval(self, i):
        self.i = i
        return i + self.value
    
    def apply_jacobian(self, J):
        return J
    
    def shape(self):
        '''Returns (size of output, size of input)'''
        n = self.value.shape[0]
        return (n, n)
    
    def self_grad(self, J):
        return J.T

        
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
    
    def apply_jacobian(self, J):
        '''
        self_J = np.zeros((self.n, self.n))
        if self.o is not None:
            for i in range(self.n):
                self_J[i,i] = np.sign(self.o[i,0])
        
        return J * self_J
        '''
        
        return np.multiply(J, np.sign(self.o).T)
    
    def shape(self):
        '''Returns (size of output, size of input)'''
        return (self.n, self.n)
    
    def self_grad(self, J):
        return None
    
class SoftMax:
    def __init__(self, n):
        self.n = n
        self.o = None
    
    def eval(self, i):
        e = np.exp(i - np.amax(i))
        self.o =  e / np.sum(e)
        return self.o
    
    def apply_jacobian(self, J):
        if self.o is None:
            return np.zeros((self.n, self.n))
        
        # Assign non-diagonal entries
        self_J = -np.multiply(self.o, self.o.T)
        # Correct diagonal entries
        di = np.diag_indices_from(self_J)
        self_J[di] = np.multiply(self.o.T, 1.-self.o.T)
        return J * self_J
    
    def shape(self):
        return (self.n, self.n)

    def self_grad(self, J):
        return None

class SimpleGradDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, op, grad):
        return -grad * self.learning_rate

class ClippedGradDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, op, grad):
        return - np.clip(grad, -1, 1) * self.learning_rate

class Momentum:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}
    
    def update(self, op, grad):
        grad = np.clip(grad, -1, 1)
        self.v[op] = self.v.get(op, 0)*self.momentum - self.learning_rate*grad
        return self.v[op]

class AdaGrad:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.r = {}
        self.e = 0.00000001

    def update(self, op, grad):
        grad = np.clip(grad, -1, 1)
        self.r[op] = self.r.get(op, 0) + np.square(grad)
        update = np.divide(self.learning_rate*grad, self.e + np.sqrt(self.r[op]))
        return -update

class RMSProp:
    def __init__(self, learning_rate, decay_rate):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.r = {}
        self.e = 0.00000001
    
    def update(self, op, grad):
        self.r[op] = (self.r.get(op, 0) * self.decay_rate
                    + np.square(grad) * (1-self.decay_rate))
        update = np.divide(self.learning_rate*grad, self.e + np.sqrt(self.r[op]))
        return -update

class RMSPropNesterov:
    def __init__(self, learning, decay, momentum):
        self.learning = learning
        self.decay = decay
        self.momentum = momentum
        self.r = {}
        self.v = {}
        self.e = 0.00000001
    
    def update(self, op, grad):
        r = self.r.get(op, 0)
        v = self.v.get(op, 0)
        self.r[op] = r * self.decay + np.square(grad) * (1-self.decay)
        vupdate = np.divide(self.learning*grad, self.e + np.sqrt(self.r[op]))
        self.v[op] = self.momentum*v - vupdate
        return self.v[op]

class Adam:
    def __init__(self, e=0.001, p1=0.9, p2=0.999):
        self.e = e
        self.p1 = p1
        self.p2 = p2
        self.s = {}
        self.r = {}
        self.d = 0.00000001
        self.t = 0
    
    def update(self, op, grad):
        s = self.s.get(op, 0)
        r = self.r.get(op, 0)
        self.t += 1
        self.s[op] = self.p1*s + (1-self.p1)*grad
        self.r[op] = self.p2*r + (1-self.p2)*np.square(grad)

        sc = self.s[op]/(1-self.p1**self.t)
        rc = self.r[op]/(1-self.p2**self.t)
        update = -np.divide(self.e*sc, self.d+np.sqrt(rc))
        
        return update

class LeastSquares:
    def cost_J(self, Yp, Y):
        return  (Yp - Y).T
    
    def error_batch(self, Yps, Ys):
        return np.sum(np.square(Yps - Ys))

class SoftMaxCrossEntropy:
    def cost_J(self, Yp, Y):
        J = -np.divide(Y, Yp+0.00000001, out=Y, where=Yp!=0.).T
        #print(J)
        return J

    def error_batch(self, Yp, Y):
        return -np.sum(np.multiply(Y, np.log(Yp, out=Yp, where=Yp!=0.)))

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
        OTHER WARNING: currently broken
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
    
    def eval_batch(self, Xs):
        return np.apply_along_axis(
            lambda x: self.eval(x.T).T,
            1,
            Xs
        )

    def backprop_batch(self, Xs, Ys, descent_obj, cost_obj):
        gradmap = {}
        for i in range(len(Xs)):
            X = np.matrix(Xs[i]).T
            Y = np.matrix(Ys[i]).T
            self.backprop_example(X, Y, gradmap, cost_obj)
        
        self.grad_update(gradmap, descent_obj)
    
    def grad_update(self, gradmap, descent_obj):
        for op, grads in gradmap.items():
            grad = sum(grads) / len(grads)
            op.value = op.value + descent_obj.update(op, grad)
        
    def backprop_example(self, X, Y, gradmap, cost_obj):
        J = cost_obj.cost_J(self.eval(X), Y)
        for op in reversed(self.operations):
            grad = op.self_grad(J)
            if grad is not None:
                if op in gradmap:
                    gradmap[op].append(grad)
                else:
                    gradmap[op] = [grad]
                
            J = op.apply_jacobian(J)

class FixedStopping:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
    
    def epochs(self, mlp, cost_obj):
        bar = progressbar.ProgressBar()
        for i in bar(range(self.num_epochs)):
            yield i
    
class EarlyStopping:
    def __init__(self, resolution, patience, max_epochs, Xs, Ys):
        self.resolution = resolution
        self.patience = patience
        self.max_epochs = max_epochs
        self.best_mlp = MLP()
        self.Xs = Xs
        self.Ys = Ys
    
    def epochs(self, mlp, cost_obj):
        i = 0
        errors = 0
        min_err = np.Infinity
        bar = progressbar.ProgressBar(max_value=self.max_epochs)
        while errors < self.patience and i < self.max_epochs:
            for j in range(self.resolution):
                bar.update(i)
                i += 1
                yield i
                

            err = cost_obj.error_batch(mlp.eval_batch(self.Xs), self.Ys)
            if err < min_err:
                errors = 0
                self.best_mlp.__dict__.update(mlp.__dict__)
                min_err = err
            else:
                errors += 1

        bar.update(i)
        mlp.__dict__ = self.best_mlp.__dict__.copy()

class MLPTrainer:
    def __init__(self, batch_size, stopping_obj, descent_obj, cost_obj):
        self.batch_size = batch_size
        self.stopping_obj = stopping_obj
        self.descent_obj = descent_obj
        self.cost_obj = cost_obj
    
    def backprop(self, Xs, Ys, mlp):
        for _ in self.stopping_obj.epochs(mlp, self.cost_obj):
            indices = np.random.permutation(len(Xs))
            initial_index = 0
            final_index = self.batch_size
            while initial_index != final_index:
                mlp.backprop_batch(
                    Xs[indices[initial_index:final_index]],
                    Ys[indices[initial_index:final_index]],
                    self.descent_obj,
                    self.cost_obj
                    )
                initial_index = final_index
                final_index = min(initial_index+self.batch_size, Xs.shape[0])


if __name__ == "__main__":
    import doctest
    doctest.testmod()