import numpy as np


class LittleModel:
    def __init__(self, X): # X has to be a matrix where every column is an input pattern
        self.W = X.T @ X
    
    def update(self, x):
        w = x @ self.W
        return np.sign(w) + (w == 0)
    
    def update_until_converge(self, x):  # x=matric with shape 1 x whatever
        num = 0
        new_x = None
        last_last_x = None
        i = 0
        while True:
            new_x = self.update(x)
            num += 1
            if (new_x == x).all(): break
            if i > 0 and (new_x == last_last_x).all(): break
            last_last_x = x
            x = new_x
            i += 1
        return new_x, num
    
    def is_attractor(self, x):
        new_x, num = self.update_until_converge(x)
        if num == 1: return True
        return False