from MLP.MLP import MLP


class Ensemble:
    def __init__(self, models: list = None):
        self.models = models

    def eval(self, X):
        sum = self.models[0].eval(X)
        for i in range(1, len(self.models)):
            sum += self.models[i].eval(X)

        return sum / len(self.models)

    def load(self, files):
        self.models = []
        for file in files:
            self.models.append(MLP())
            self.models[-1].load(file)
