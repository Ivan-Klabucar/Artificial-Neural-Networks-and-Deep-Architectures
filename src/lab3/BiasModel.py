import numpy as np


class BiasModel:
    def __init__(self, X):
        self.ro = np.sum(X) / (X.shape[0] * X.shape[1])
        self.W = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                for k in range(X.shape[0]):
                    self.W[i][j] += (X[k][i] - self.ro) * (X[k][j] - self.ro)

    def update(self, x, bias):
        w = x @ self.W - bias
        return 0.5 + 0.5 * (np.sign(w) + (w == 0))

    def update_until_converge(self, x, bias):  # x=matric with shape 1 x whatever
        num = 0
        new_x = None
        while True:
            new_x = self.update(x, bias)
            num += 1
            if (new_x == x).all():
                break
            x = new_x
        return new_x, num

    def is_attractor(self, x, bias):
        new_x, num = self.update_until_converge(x, bias)
        if num == 1:
            return True
        return False
