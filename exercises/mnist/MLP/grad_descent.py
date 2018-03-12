import numpy.matlib as np


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