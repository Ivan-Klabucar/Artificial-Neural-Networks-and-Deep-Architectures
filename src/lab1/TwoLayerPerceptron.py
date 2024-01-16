import numpy as np

def F_phi(x):
    return 2/(1 + np.exp(-x)) - 1

def dF_phi(x):
    return ((1 + F_phi(x))*(1 - F_phi(x)))/2

class TwoLayerPerceptron:
    """
    A class used to represent a Two Layer Perceptron

    Attributes
    ----------
    V:
        weights between hidden layer and output layer
    W:
        weights between inputs and hidden layer

    eta:
        learning rate

    alfa:
        momentum rate alfa=0 => no momentum contribution

    theta,psi:
        accumulator variables for momentum, unimportant when alfa = 0

    Methods
    -------
    forward_propagation

    epoch
    """

    def __init__(self, n_in, n_h, n_out, eta, alfa=0.0):
        self.eta = eta
        self.alfa = alfa
        self.W = np.random.normal(loc=0.0, scale=np.sqrt(1/(n_in + 1)), size=(n_h, n_in + 1))
        self.V = np.random.normal(loc=0.0, scale=np.sqrt(1/(n_h + 1)), size=(n_out, n_h + 1))
        self.theta = 0
        self.psi = 0

    def M_forwardPropagation(self, X):
        """Forward propagation in the perceptron

        Parameters
        ----------
        X: numpy array (number of features, number of datapoints)
            observation data

        Return
        ------
        y: numpy array (number of output neurons, number of datapoints)
        """
        X = np.append(X, np.ones((1, X.shape[1])), axis=0)
        self.H_star = self.W @ X
        self.H = F_phi(self.H_star)
        self.H = np.append(self.H, np.ones((1, self.H.shape[1])), axis=0)
        self.O_star = self.V @ self.H
        self.O = F_phi(self.O_star)
        return self.O

    def M_backwardPropagation(self, X, T):
        """Backward propagation in the MLP

        Parameters
        ----------
        X: numpy array (number of features, number of datapoints)
            observation data
        T: numpy array (number of output neurons, number of datapoints)
            targets

        """
        X = np.append(X, np.ones((1, X.shape[1])), axis=0)
        delta_o = (self.O - T) * dF_phi(self.O_star)
        delta_h = (self.V.T[:-1] @ delta_o) * dF_phi(self.H_star)
        self.theta = self.alfa * self.theta - (1 - self.alfa) * delta_h @ X.T
        self.psi = self.alfa * self.psi - (1 - self.alfa) * delta_o @ self.H.T
        self.W += self.eta * self.theta
        self.V += self.eta * self.psi

def small_test():
    net = TwoLayerPerceptron(2,3,2, 0.1)
    X = np.array([[5,2], [6,3], [4,8], [4,5], [0,1], [1,0], [0,0], [-1,0]]).T
    T = np.array([[1,0], [1,0], [1,0], [1,0], [0,1], [0,1], [0,1], [0,1]]).T
    print(T)
    for i in range(100):
        out = net.M_forwardPropagation(X)
        print(out)
        print()
        net.M_backwardPropagation(X, T)

def small_test2():
    net = TwoLayerPerceptron(2,3,1, 0.1, 0.01)
    X = np.array([[5,2], [6,3], [4,8], [4,5], [0,1], [1,0], [0,0], [-1,0]]).T
    T = np.array([[1], [1], [1], [1], [0], [0], [0], [0]]).T
    print(T)
    for i in range(100):
        out = net.M_forwardPropagation(X)
        print(out)
        print()
        net.M_backwardPropagation(X, T)

# small_test2()