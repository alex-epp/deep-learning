import numpy.matlib as np
import pickle

from .op import MatMul

class L2Regularization:
    def __init__(self, _lambda):
        self._lambda = _lambda

    def update(self, op):
        if isinstance(op, MatMul):
            return - self._lambda * op.value
        else:
            return 0

class FixedStopping:
    def __init__(self, num_epochs):
        self.max_epochs = num_epochs

    def epochs(self, mlp, cost_obj):
        for i in range(self.max_epochs):
            yield i


class EarlyStopping:
    def __init__(self, resolution, patience, max_epochs, Xs, Ys):
        self.resolution = resolution
        self.patience = patience
        self.max_epochs = max_epochs
        self.best_mlp = MLP()
        self.Xs = np.matrix(Xs, dtype=np.float32)
        self.Ys = np.matrix(Ys, dtype=np.float32)

    def epochs(self, mlp, cost_obj):
        i = 0
        errors = 0
        min_err = np.Infinity
        while errors < self.patience and i < self.max_epochs:
            for j in range(self.resolution):
                i += 1
                yield i


            err = cost_obj.error_batch(mlp.eval_batch(self.Xs), self.Ys)
            if err < min_err:
                errors = 0
                self.best_mlp.__dict__.update(mlp.__dict__)
                min_err = err
            else:
                errors += 1

        mlp.__dict__ = self.best_mlp.__dict__.copy()


class MLP:
    def __init__(self):
        self.ops = []

    def add_op(self, op):
        self.ops.append(op)

    def eval(self, X):
        out = np.matrix(X, dtype=np.float32, copy=False)

        for op in self.ops:
            out = op.eval(out)

        return out

    def eval_batch(self, Xs):
        out = np.matrix(Xs, dtype=np.float32, copy=False)

        return np.apply_along_axis(
            lambda x: self.eval(x.T).T,
            1,
            Xs
        )

    def save(self, file):
        pickle.dump(self.ops, file)

    def load(self, file):
        self.ops = pickle.load(file)


class MLPTrainer:
    def __init__(self, batch_size, stopping_obj, descent_obj, cost_obj, visual_obj, reg_obj=None):
        self.batch_size = batch_size
        self.stopping_obj = stopping_obj
        self.descent_obj = descent_obj
        self.cost_obj = cost_obj
        self.visual_obj = visual_obj
        self.reg_obj = reg_obj

    def backprop(self, Xs, Ys, mlp):
        Xs = np.matrix(Xs, dtype=np.float32, copy=False)
        Ys = np.matrix(Ys, dtype=np.float32, copy=False)

        self.visual_obj.start(
            self.stopping_obj.max_epochs,
            int(len(Xs)/self.batch_size)
        )

        for _ in self.stopping_obj.epochs(mlp, self.cost_obj):
            indices = np.random.permutation(len(Xs))
            initial_index = 0
            final_index = self.batch_size
            while initial_index != final_index:
                self.backprop_minibatch(
                    mlp,
                    Xs[indices[initial_index:final_index]],
                    Ys[indices[initial_index:final_index]],
                    )
                initial_index = final_index
                final_index = min(initial_index+self.batch_size, Xs.shape[0])

                self.visual_obj.update_minibatch(mlp)

            self.visual_obj.update_epoch(mlp)

    def backprop_minibatch(self, mlp, Xs, Ys):
        minibatch_size = len(Xs)
        grad_sums = [None for _ in range(len(mlp.ops))]

        for i in range(minibatch_size):
            X = np.matrix(Xs[i]).T
            Y = np.matrix(Ys[i]).T
            self.backprop_example(mlp, X, Y, grad_sums)

        for i in range(len(mlp.ops)):
            if grad_sums[i] is not None:
                grad = grad_sums[i] / minibatch_size
                if self.reg_obj is not None:
                    mlp.ops[i].update(self.reg_obj.update(mlp.ops[i]))

                mlp.ops[i].update(self.descent_obj.update(mlp.ops[i], grad))

    def backprop_example(self, mlp, X, Y, grad_sums):
        grad_upstream = self.cost_obj.grad(mlp.eval(X), Y)
        for i in reversed(range(len(mlp.ops))):
            grad = mlp.ops[i].grad_self(grad_upstream)
            if grad is not None:
                if grad_sums[i] is not None:
                    grad_sums[i] += grad
                else:
                    grad_sums[i] = np.copy(grad)

            grad_upstream = mlp.ops[i].backprop(grad_upstream)
