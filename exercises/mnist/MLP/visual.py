import numpy.matlib as np
import matplotlib.pyplot as plt
try:
    import progressbar
except ImportError:
    print("Error--cannot import progressbar2: MLP.visual will not work")


class BarVisualizer:
    def __init__(self):
        self.bar = None
        self.pos = None

    def start(self, num_epochs, num_minibatches):
        self.bar = progressbar.ProgressBar(max_value=num_epochs*num_minibatches)
        self.pos = 0
        self.bar.update(self.pos)

    def update_minibatch(self, mlp):
        self.pos += 1
        self.bar.update(self.pos)

    def update_epoch(self, mlp):
        pass

class GraphVisualizer:
    def __init__(self, tr_Xs, tr_Ys, va_Xs, va_Ys, cost_obj):
        self.pos = None
        self.tr_Xs = np.matrix(tr_Xs, dtype=np.float32)
        self.tr_Ys = np.matrix(tr_Ys, dtype=np.float32)
        self.va_Xs = np.matrix(va_Xs, dtype=np.float32)
        self.va_Ys = np.matrix(va_Ys, dtype=np.float32)
        self.cost_obj = cost_obj
        self.bar = BarVisualizer()

    def start(self, num_epochs, num_minibatches):
        self.pos = 0
        plt.ion()
        plt.show(block=False)

        self.bar.start(num_epochs, num_minibatches)

    def update_minibatch(self, mlp):
        self.bar.update_minibatch(mlp)
        plt.pause(.00005)

    def update_epoch(self, mlp):
        self.bar.update_epoch(mlp)

        tr_err = self.cost_obj.error_batch(
            mlp.eval_batch(self.tr_Xs),
            self.tr_Ys,
        )  / self.tr_Xs.shape[0]
        plt.scatter(self.pos, tr_err, c='b')

        va_err = self.cost_obj.error_batch(
            mlp.eval_batch(self.va_Xs),
            self.va_Ys,
        )  / self.va_Xs.shape[0]
        plt.scatter(self.pos, va_err, c='r')

        self.pos += 1
