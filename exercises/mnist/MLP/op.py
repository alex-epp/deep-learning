import numpy.matlib as np


class MatMul:
    def __init__(self, weights):
        self.value = np.matrix(weights, dtype=np.float32)
        self.last_input = 0
        
    def eval(self, i):
        self.last_input = i
        return self.value*i
    
    def backprop(self, upstream_grad):
        return upstream_grad * self.value
    
    def grad_self(self, upstream_grad):
        return np.multiply(upstream_grad, self.last_input).T
    
    def update(self, diff):
        self.value += diff


class VecAdd:
    def __init__(self, bias):
        self.value = np.matrix(bias, dtype=np.float32)
        self.last_input = 0
    
    def eval(self, i):
        self.last_input = i
        return i + self.value
    
    def backprop(self, upstream_grad):
        return upstream_grad
    
    def grad_self(self, upstream_grad):
        return upstream_grad.T
    
    def update(self, diff):
        self.value += diff

        
class Rectify:
    def __init__(self, n):
        self.last_output = 0
        self.last_input = 0
        self.n = n
    
    def eval(self, i):
        self.last_input = i
        self.last_output = np.maximum(0, i)
        return self.last_output
    
    def backprop(self, upstream_grad):
        return np.multiply(upstream_grad, np.sign(self.last_output).T)
    
    def grad_self(self, upstream_grad):
        return None


class SoftMax:
    def __init__(self, n):
        self.n = n
        self.last_output = 0
    
    def eval(self, i):
        e = np.exp(i - np.amax(i))
        self.last_output =  e / np.sum(e)
        return self.last_output
    
    def backprop(self, upstream_grad):        
        # Assign non-diagonal entries
        self_J = -np.multiply(self.last_output, self.last_output.T)
        # Correct diagonal entries
        di = np.diag_indices_from(self_J)
        self_J[di] = np.multiply(self.last_output.T, 1.-self.last_output.T)
        return upstream_grad * self_J
    
    def shape(self):
        return (self.n, self.n)

    def grad_self(self, J):
        return None
    