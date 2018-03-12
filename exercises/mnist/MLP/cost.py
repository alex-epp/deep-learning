import numpy.matlib as np


class LeastSquares:
    def grad(self, Yp, Y):
        return  (Yp - Y).T
    
    def error_batch(self, Yps, Ys):
        return np.sum(np.square(Yps - Ys))

class SoftMaxCrossEntropy:
    def grad(self, Yp, Y):
        J = -np.divide(Y, Yp+0.00000001, out=Y, where=Yp!=0.).T
        #print(J)
        return J

    def error_batch(self, Yp, Y):
        return -np.sum(np.multiply(Y, np.log(Yp, out=Yp, where=Yp!=0.)))
