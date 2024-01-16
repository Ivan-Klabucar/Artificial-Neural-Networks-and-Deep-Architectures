#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator

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
    labels = np.array([0] * n + [1] * n)

    #Shuffleling
    indices = np.arange(2 * n)
    np.random.shuffle(indices)
    data_shuffle, labels_shuffle = data[:, indices], labels[indices]

    return data_shuffle, labels_shuffle


def dataSubsampler(data, labels, p1, p2): #Keeps p1 percent of the first class and p2 percent of the second class
    indices = np.arange(0, labels.shape[0])
    classAIndices = indices[labels == 0]
    classBIndices = indices[labels == 1]

    maskA = np.random.rand(classAIndices.shape[0]) < p1
    maskB = np.random.rand(classBIndices.shape[0]) < p2

    classAIndices = classAIndices[maskA]
    classBIndices = classBIndices[maskB]
    indices = np.concatenate((classAIndices, classBIndices))

    data = data[:, indices]
    labels = labels[indices]

    #Shuffleling
    indices = np.arange(indices.shape[0])
    np.random.shuffle(indices)
    data_shuffle, labels_shuffle = data[:, indices], labels[indices]

    return data_shuffle, labels_shuffle


n = 300
x_shuffle, y_shuffle = dataCloudGenerator(n)
print(y_shuffle.shape)

x_train, x_test, y_train, y_test = train_test_split(x_shuffle.T, y_shuffle, test_size=0.3)
x_train, x_test = x_train.T, x_test.T
x_train_subample, y_train_subsample = dataSubsampler(x_train, y_train, 0.8, 0.2)

plt.scatter(x_train_subample[0], x_train_subample[1], c=colormap[y_train_subsample])
plt.show()
# %% Perceptron

class C_Perceptron:
    """
    A class used to represent a Perceptron

    ...

    Attributes
    ----------
    W:
        weights and biases to be learnt
 
    Methods
    -------
    forward_propagation
    
    epoch
    """

    def __init__(self, n_in, n_out, eta, batch_mode = True, delta_rule=True):
        #initialise weight and biases parameters
        self.W = np.random.randn(n_out, n_in + 1) * 0.1
        self.eta = eta
        self.batch_mode = batch_mode
        self.delta_rule = delta_rule
        return
            

    def M_predict(self, X):
        """Forward propagation in the perceptron

        Parameters
        ----------
        X: numpy array (nbDim, nbData)
            observation data

        Return
        ------
        y: numpy array (1, nbData)
            predicted value by the perceptron
        """
        X = np.append(X, np.ones((1, X.shape[1])), axis=0)
        return self.W @ X


    def M_train(self, X, hat_y):
        """Updates the weights and bias of the perceptron

        Parameters
        ----------
        X: numpy array (nbDim, nbData)
            observation data
        hat_y: numpy array (1, nbData)
            ground-truth class to predict (0 and 1)
            
        """
        X = np.append(X, np.ones((1, X.shape[1])), axis=0)

        if self.batch_mode:
            if self.delta_rule:
                truth = (hat_y - 0.5) * 2
                self.W -= self.eta * (self.W @ X - truth) @ X.T

            else:
                predictions = self.W @ X > 0
                self.W -= self.eta * (predictions - hat_y) @ X.T

        else:
            index = np.random.randint(0, X.shape[1])
            sample = X[:, index]
            self.W -= self.eta * (self.W @ sample - hat_y[index]) * sample

        return self.W

    

#Train the perceptron
perceptronDelta = C_Perceptron(2, 1, 0.001, batch_mode=True, delta_rule=True)
perceptronSimple = C_Perceptron(2, 1, 0.001, batch_mode=True, delta_rule=False)
accuracy1 = [0]
accuracy2 = [0]

x_lin = np.linspace(-1, 1, 100)
plt.ion() # turn interactive mode on
fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.scatter(x_train_subample[0], x_train_subample[1], c=colormap[y_train_subsample])
ax.scatter(x_train[0], x_train[1], c=colormap[y_train], marker=".")
line1, = ax.plot(x_lin, x_lin, label="Delta learning rule")
# line2, = ax.plot(x_lin, x_lin, label="Perceptron learning rule")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Decision boundary of the perceptron")

fig.canvas.draw()
plt.pause(0.1)


for i in range(20):
    W1 = perceptronDelta.M_train(x_train_subample, y_train_subsample)[0]
    W2 = perceptronSimple.M_train(x_train_subample, y_train_subsample)[0]
    
    line1.set_ydata(-W1[0] / W1[1] * x_lin - W1[2] / W1[1])
    # line2.set_ydata(-W2[0] / W2[1] * x_lin - W2[2] / W2[1])
    fig.canvas.draw()
    fig.canvas.flush_events()

    predictions = perceptronDelta.M_predict(x_train_subample)[0] > 0.
    accuracy1.append(accuracy_score(y_train_subsample, predictions))
    
    predictions = perceptronSimple.M_predict(x_train_subample)[0] > 0.
    accuracy2.append(accuracy_score(y_train_subsample, predictions))

    # predictions = perceptron1.M_predict(x_test)[0] > 0.
    # testing_accuracy.append(accuracy_score(y_test, predictions))
plt.ioff()

predictions = perceptronDelta.M_predict(x_test)[0] > 0.
print(f"Accuracy test = {accuracy_score(y_test, predictions)}")
cm = confusion_matrix(y_test, predictions) #, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot() 
plt.show()

plt.title("Accuracy")
plt.ylim([0.5, 1.1])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.xticks(np.arange(0, 40, 5)) #.set_major_locator(MaxNLocator(integer=True))
plt.plot(accuracy1, label='Training perceptron rule')
plt.plot(accuracy2, label='Training delta rule')
plt.legend()
plt.show()

# %%
