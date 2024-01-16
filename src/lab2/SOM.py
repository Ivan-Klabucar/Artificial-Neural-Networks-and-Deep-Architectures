#%% Imports
import numpy as np
import matplotlib.pyplot as plt

#%% The SOM algorithm

def create_circular_mask(h, w, center=None, radius=None):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


class SOM:

    def __init__(self, n_in, n_out, initial_radius=1, radius_decay=0.5, eta=1.1):
        
        self.W = (np.random.uniform(size=n_out + [n_in]) - 0.5) * 10
        self.initial_radius = initial_radius
        self.radius_decay = radius_decay
        self.eta = eta
        self.epoch = 0
        return

    def training(self, X):
        """Training step of the SOM

        Parameters
        ----------
        X: numpy array (n_in)
            observation data
        
        """

        #Find BMU
        distances = np.sum(self.W - X, axis=-1)
        BMU = np.unravel_index(np.argmax(distances), distances.shape)
        print()

        radius = self.initial_radius * np.exp(- self.epoch * self.radius_decay)
        circle_mask = create_circular_mask(self.W.shape[0], self.W.shape[1], BMU, radius)
        circle_mask = np.repeat(circle_mask[:, :, np.newaxis], 2, axis=2 )
        
        #Parameters update
        # print((self.eta * (self.W - X) * circle_mask)[:,:,0])

        self.W = self.W - self.eta * (self.W - X) * circle_mask

        return self.W

    def train_epoch(self, X):
        """Training step of the SOM

        Parameters
        ----------
        X: numpy array (n_in, nbData)
            observation data
        
        """

        indices = np.arange(0, X.shape[1])
        np.random.shuffle(indices)
        for i in indices:
            self.training(X[:, i])


        #Update epoch
        self.epoch += 1

    
    def getFlattenW(self):
        flattenW = np.array([self.W[:, :, 0].flatten(), self.W[:, :, 1].flatten()])
        return flattenW


#%% Import & Preprocess the data
data1 = np.random.uniform(size=[2, 200]) - 0.5
data2 = np.random.uniform(size=[2, 200]) - 0.5
data1[0] *= 10
data2[1] *= 10
data = np.concatenate((data1, data2), axis=1)
plt.scatter(data1[0], data1[1])
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.show()



#%% Training the SOM
SOMnet = SOM(2, [10, 10], initial_radius=5, radius_decay=0., eta=0.001)


plt.ion() # turn interactive mode on
fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.scatter(data1[0], data1[1])

flattenW = SOMnet.getFlattenW()
sc = ax.scatter(flattenW[0], flattenW[1])

plt.xlabel("x")
plt.ylabel("y")
plt.title("Decision boundary of the perceptron")

fig.canvas.draw()
plt.pause(1)

for i in range(10):
    SOMnet.train_epoch(data1)
    
    flattenW = SOMnet.getFlattenW()
    sc.set_offsets(flattenW.T) 
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(1)


plt.ioff()
# %%
