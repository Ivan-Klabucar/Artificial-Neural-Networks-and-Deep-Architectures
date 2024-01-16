import numpy as np

# Simple Hopfield network with sequential update,
# if scale_weights=True, self.W is multiplied by 1/N, N = number of units, no bias
# x=matric with shape 1 x whatever


class AsyncHopfield:
    def __init__(self, X, scale_weights=False, update_listener=None, limit=None): # X has to be a matrix where every column is an input pattern
        self.W = X.T @ X
        self.number_of_update = None
        self.update_listener = update_listener
        self.limit = limit
        if scale_weights: self.W = (1/self.W.shape[0]) * self.W
    
    def update_epoch(self, x): # x=matrix with shape 1 x whatever
        mutating_x = x.copy()
        rand_order = np.arange(x.shape[1])
        np.random.shuffle(rand_order)
        for i in rand_order:
            new_xi = np.sign(np.dot(mutating_x[0], self.W[:,i]))
            if new_xi == 0: new_xi = 1
            mutating_x[0][i] = new_xi
            self.number_of_update += 1
            if self.update_listener: self.update_listener.update(mutating_x, self.number_of_update, self)
        converged = (mutating_x == x).all()
        return mutating_x, converged
    
    def update_until_converge(self, x): # x=matric with shape 1 x whatever
        num = 0
        self.number_of_update = 0
        mutating_x, converged = self.update_epoch(x)
        num += 1
        while not converged:
            mutating_x, converged = self.update_epoch(mutating_x)
            num += 1
            if self.limit and num >= self.limit: break
        self.number_of_update = None
        return mutating_x, num

    def is_attractor(self, x):
        new_x, num = self.update_until_converge(x)
        if num == 1: return True
        return False
    
    def energy(self, x):
        x_star = x @ self.W
        return -np.dot(x[0], x_star[0])