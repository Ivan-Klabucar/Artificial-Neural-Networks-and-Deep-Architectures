import numpy as np

def distance(x, y):
    return np.linalg.norm(x - y)

def rb_function(center, x, sigma):
    return np.exp((-distance(center, x)**2)/(2*sigma**2))

class RBF:
    def __init__(self, input_dim, sigma):
        self.sigma = sigma
        self.input_dim = input_dim
    
    def set_nodes(self, nodes):
        if not len(nodes.shape) == 2: raise "Nodes need to be a 2d matrix"
        if not nodes.shape[1] == self.input_dim: 
            raise "Node dimesnsion different from input dimesnsion"

        self.nodes = nodes
    
    def phi(self, input):
        phi = []
        for pattern in input:
            row = []
            for node in self.nodes:
                row.append(rb_function(node, pattern, self.sigma))
            phi.append(row)
        return np.array(phi)
    
    def get_output_dim(self):
        return self.nodes.shape[0]
            

                


        

