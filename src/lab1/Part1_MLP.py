#%% Imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

colormap = np.array(['r', 'g'])

#%% Generation of linearly separable data
linear = False
if linear:
    #Center for linearly separable data
    Acenter = [1.5, 0.5]
    Bcenter = [-1.5, 0.5]
else:
    #Center for non linearly separable data
    Acenter = [0.75, 0.5]
    Bcenter = [-0.75, 0.5]

def dataCloudGenerator (n):
    mA, sigmaA = np.array(Acenter), 0.5
    mB, sigmaB = np.array(Bcenter), 0.5


    classA = (np.random.randn(n, 2) * sigmaA + mA).T
    classB = (np.random.randn(n, 2) * sigmaB + mB).T

    data = np.concatenate((classA, classB), axis=1)
    labels = np.array([[0] * n + [1] * n])
    print(labels.shape)

    #Shuffleling
    indices = np.arange(2 * n)
    np.random.shuffle(indices)
    data_shuffle, labels_shuffle = data[:, indices], labels[:, indices]

    return data_shuffle, labels_shuffle

def dataCirclesGenerator(n):
    data, labels = make_circles(n, noise=0.25, factor=0.4)

    return data.T, np.array([labels])

def dataSubsampler(data, labels, p1, p2): #Keeps p1 percent of the first class and p2 percent of the second class
    indices = np.arange(0, labels.shape[1])
    classAIndices = indices[labels[0] == 0]
    classBIndices = indices[labels[0] == 1]

    maskA = np.random.rand(classAIndices.shape[0]) < p1
    maskB = np.random.rand(classBIndices.shape[0]) < p2

    classAIndices = classAIndices[maskA]
    classBIndices = classBIndices[maskB]
    indices = np.concatenate((classAIndices, classBIndices))

    data = data[:, indices]
    labels = labels[:, indices]

    #Shuffleling
    indices = np.arange(indices.shape[0])
    np.random.shuffle(indices)
    data_shuffle, labels_shuffle = data[:, indices], labels[:, indices]

    return data_shuffle, labels_shuffle


n = 200
x_shuffle, y_shuffle = dataCirclesGenerator(n)

x_train, x_test, y_train, y_test = train_test_split(x_shuffle.T, y_shuffle.T, test_size=0.1)
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
x_train_subample, y_train_subsample = dataSubsampler(x_train, y_train, 1, 1)

plt.scatter(x_train[0], x_train[1], c=colormap[y_train[0]])
plt.scatter(x_test[0], x_test[1], c=colormap[y_test[0]], marker=".")
plt.show()


#%% MLP
def F_sigmoid(x):
    """Compute the value of the sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def F_relu(x):
    """Compute the value of the Rectified Linear Unit activation function"""
    return x * (x > 0)

def F_dRelu(x):
    """Compute the derivative of the Rectified Linear Unit activation function"""
    y = x
    x[x<=0] = 0
    x[x>0] = 1
    return x

def F_dSigmoid(x):
    """Compute the derivative of the Rectified Linear Unit activation function"""
    return ((1 + F_sigmoid(x))*(1 - F_sigmoid(x)))/2

def F_computeCost(y,hat_y):
    """Compute the cost (sum of the losses)
    
    Parameters
    ----------
    y: (1, nbData)
        predicted value by the MLP
    hat_y: (1, nbData)
        ground-truth class to predict
    """
    m = y.shape[1]
     
    # --- START CODE HERE
    loss = -hat_y * np.log(y) - (1-hat_y)* np.log(1-y)
    # --- END CODE HERE
    
    cost = np.sum(loss) / m
    return cost

def F_computeAccuracy(y,hat_y):
    """Compute the accuracy
    
    Parameters
    ----------
    y: (nbData)
        predicted value by the MLP
    hat_y: (nbData)
        ground-truth class to predict
    """
    
    m = hat_y.shape[1]    
    class_y = np.copy(y)
    class_y[class_y>=0.5]=1
    class_y[class_y<0.5]=0
    return np.sum(class_y==hat_y) / m

class C_MultiLayerPerceptron:
    """
    A class used to represent a Multi-Layer Perceptron with 1 hidden layers

    ...

    Attributes
    ----------
    W1, b1, W2, b2:
        weights and biases to be learnt
    Z1, A1, Z2, A2:
        values of the internal neurons to be used for backpropagation
    dW1, db1, dW2, db2, dZ1, dZ2:
        partial derivatives of the loss w.r.t. parameters
        
    Methods
    -------
    forward_propagation
    
    backward_propagation
    
    update_parameters
    
    """

    W1, b1, W2, b2 = [], [], [], []
    Z1, A1, Z2, A2 = [], [], [], []
    dW1, db1, dW2, db2 = [], [], [], []   
    dZ1, dA1, dZ2 = [], [], []
    
    def __init__(self, n_in, n_h, n_out):
        #initialise weight and biases parameters
        self.W1 = np.random.randn(n_h, n_in) * 0.01
        self.b1 = np.zeros(shape=(n_h, 1))
        self.W2 = np.random.randn(n_out, n_h) * 0.01
        self.b2 = np.zeros(shape=(n_out, 1))
        return

    
    def __setattr__(self, attrName, val):
        if hasattr(self, attrName):
            self.__dict__[attrName] = val
        else:
            raise Exception("self.%s note part of the fields" % attrName)

            

    def M_forwardPropagation(self, X):
        """Forward propagation in the MLP

        Parameters
        ----------
        X: numpy array (nbDim, nbData)
            observation data

        Return
        ------
        y: numpy array (nbData)
            predicted value by the MLP
        """
        
        # --- START CODE HERE 
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = F_relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = F_sigmoid(self.Z2)
        # --- END CODE HERE
        
        y = self.A2
        
        return y


    def M_backwardPropagation(self, X, hat_y):
        """Backward propagation in the MLP

        Parameters
        ----------
        X: numpy array (nbDim, nbData)
            observation data
        hat_y: numpy array (nbData)
            ground-truth class to predict
            
        """
        
        m = hat_y.shape[1]  #batch size
        
        # --- START CODE HERE
        self.dZ2 = self.A2 - hat_y       
        self.dW2 = (self.dZ2 @ self.A1.T ) / m
        self.db2 = np.sum(self.dZ2) / m
        self.dA1 = self.W2.T @ self.dZ2
        self.dZ1 = self.dA1 * F_dRelu(self.Z1)
        self.dW1 = self.dZ1 @ X.T / m
        self.db1 = np.sum(self.dZ1) / m
        # --- END CODE HERE
        return

    
    def M_gradientDescent(self, alpha):
        """Update the parameters of the network using gradient descent

        Parameters
        ----------
        alpha: float scalar
            amount of update at each step of the gradient descent
            
        """

        # --- START CODE HERE
        self.W1 = self.W1 - alpha * self.dW1
        self.b1 = self.b1 - alpha * self.db1
        self.W2 = self.W2 - alpha * self.dW2
        self.b2 = self.b2 - alpha * self.db2
        # --- END CODE HERE
        
        return


#%% First graph

plt.title("Accuracy during training for everal network sizes")

for idx, nbNode in enumerate([5, 20, 100, 1000]):
    net = C_MultiLayerPerceptron(2, n, 1)
    train_cost, train_accuracy, test_cost, test_accuracy = [], [], [], []
    for i in range(4000):
        y_predict_train = net.M_forwardPropagation(x_train)
        
        # --- Store results on train
        train_cost.append( F_computeCost(y_predict_train, y_train) )
        train_accuracy.append( F_computeAccuracy(y_predict_train, y_train) )
        
        # --- Backward
        net.M_backwardPropagation(x_train, y_train)
        
        # --- Update
        net.M_gradientDescent(alpha=0.05)

        # --- Store results on test
        y_predict_test = net.M_forwardPropagation(x_test)
        test_cost.append( F_computeCost(y_predict_test, y_test) )    
        test_accuracy.append( F_computeAccuracy(y_predict_test, y_test) )

    plt.plot(train_accuracy, label=f"Accuracy with {nbNode} hidden nodes")

plt.legend()
plt.ylim([0.4, 1.1])
plt.show()

#%% second graph


# net = C_MultiLayerPerceptron(2, 1000, 1)
# train_cost, train_accuracy, test_cost, test_accuracy = [], [], [], []
# for i in range(20000):
#     y_predict_train = net.M_forwardPropagation(x_train)
    
#     # --- Store results on train
#     train_cost.append( F_computeCost(y_predict_train, y_train) )
#     train_accuracy.append( F_computeAccuracy(y_predict_train, y_train) )
    
#     # --- Backward
#     net.M_backwardPropagation(x_train, y_train)
    
#     # --- Update
#     net.M_gradientDescent(alpha=0.05)

#     # --- Store results on test
#     y_predict_test = net.M_forwardPropagation(x_test)
#     test_cost.append( F_computeCost(y_predict_test, y_test) )    
#     test_accuracy.append( F_computeAccuracy(y_predict_test, y_test) )

# plt.title("Accuracy during training for everal network sizes")
# plt.plot(train_accuracy, label=f"Train accuracy with 500 hidden nodes")
# plt.plot(test_accuracy, label=f"Test accuracy with 500 hidden nodes")
# plt.legend()
# plt.ylim([0.4, 1.1])
# plt.show()

# # define bounds of the domain
# min1, max1 = x_train[0, :].min()-1, x_train[0, :].max()+1
# min2, max2 = x_train[1, :].min()-1, x_train[1, :].max()+1

# # define the x and y scale
# x1grid = np.arange(min1, max1, 0.05)
# x2grid = np.arange(min2, max2, 0.05)

# # create all of the lines and rows of the grid
# xx, yy = np.meshgrid(x1grid, x2grid)

# # flatten each grid to a vector
# r1, r2 = xx.flatten(), yy.flatten()
# r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

# # horizontal stack vectors to create x1,x2 input for the model
# grid = np.hstack((r1,r2))

# # make predictions for the grid
# yhat = net.M_forwardPropagation(grid.T)
# yhat = yhat > 0.5
# # reshape the predictions back into a grid
# zz = yhat.reshape(xx.shape)

# # plot the grid of x, y and z values as a surface
# plt.contourf(xx, yy, zz, cmap='Paired')
# plt.scatter(x_train[0], x_train[1], c=colormap[y_train[0]])
# plt.scatter(x_test[0], x_test[1], c=colormap[y_test[0]], marker=".")
# plt.show()


# %%Second graph pytorch