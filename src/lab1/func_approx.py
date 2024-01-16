#%% Imports
from logging import error
import numpy as np
import matplotlib.pyplot as plt
from TwoLayerPerceptron import TwoLayerPerceptron

def gauss(x, y):
    return np.exp(-x*x*0.1) * np.exp(-y*y*0.1) - 0.5

def sq_err(x, y):
    tmp = x - y
    tmp = tmp * tmp
    return np.sum(tmp)

def mean_sq_err(x, y):
    return sq_err(x, y)/x.size

#%% Generation of datapoints and visualization
x_points = np.linspace(-5, 5, 21)
y_points = np.linspace(-5, 5, 21)
X, Y = np.meshgrid(x_points, y_points)
# gauss_2d = np.vectorize(gauss)
Z = gauss(X, Y)

plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='terrain', edgecolor=None)
ax.set(xlabel='x', ylabel='y', zlabel='f(x, y)', title='Function to be approximated')

input = np.dstack([X, Y]).reshape(-1, 2).T
T = Z.reshape(1,-1)

#%% Training of the first MLP
net = TwoLayerPerceptron(2,10,1, 0.005)
out = []

fig = plt.figure(2)
ax = plt.axes()
err = []
for i in range(2000):
    out = net.M_forwardPropagation(input)
    me = mean_sq_err(out, T)
    err.append(me)
    net.M_backwardPropagation(input, T)
ax.plot(np.linspace(0, 2000, 2000), err)
outr = out.reshape(21, 21)
tr = T.reshape(21, 21)

plt.figure(3)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, outr, rstride=1, cstride=1, cmap='terrain', edgecolor=None)
ax.set(xlabel='x', ylabel='y', zlabel='f(x, y)', title='Function approximated bv MLP 100/0 split')


#%% Spliting the data into validation and training
patterns_and_targets = np.append(input, T, axis=0)
np.random.shuffle(patterns_and_targets.T)
total = patterns_and_targets.shape[1]
num_for_training = int(total * 0.8)
training_data = patterns_and_targets.T[0:num_for_training].T
validation_data = patterns_and_targets.T[num_for_training:].T
input_train = training_data[0:2]
T_train = training_data[2:]
input_validation = validation_data[0:2]
T_validation = validation_data[2:]

num_of_hnodes = [1, 3, 5, 10, 25]
errors = []
nf = 4
for n in num_of_hnodes:
    net = TwoLayerPerceptron(2, n, 1, 0.003)
    err = []
    v_err = []
    for i in range(1500):
        v_out = net.M_forwardPropagation(input_validation)
        out = net.M_forwardPropagation(input_train)
        mse = mean_sq_err(out, T_train)
        v_mse = mean_sq_err(v_out, T_validation)
        err.append(mse)
        v_err.append(v_mse)
        net.M_backwardPropagation(input_train, T_train)
    fig = plt.figure(nf)
    ax = plt.axes()
    ax.plot(range(1500), err, color='cornflowerblue', label='training error')
    print(f'verr: {len(v_err)}')
    ax.plot(range(1500), v_err, color='indianred', label='validation error')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    plt.title(f'Error during training architecture 2x{n}')

    fig = plt.figure(nf+1)
    nf += 2
    ax = fig.add_subplot(projection='3d')
    ax.scatter(input_train[0], input_train[1], out[0], marker='o')
    val_out = net.M_forwardPropagation(input_validation)
    ax.scatter(input_validation[0], input_validation[1], val_out[0], marker='o', color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.legend(['training', 'validation'])
    plt.title(f'Predictions of architecture 2x{n}')
    print(f'Error on training data architecture 2x{n}: {mean_sq_err(out, T_train)}')
    print(f'Error on validation data architecture 2x{n}: {mean_sq_err(val_out, T_validation)}')
    plt.show()

mean_err = []
std_err = []
for n in num_of_hnodes:
    v_errs = []
    for i in range(100):
        net = TwoLayerPerceptron(2, n, 1, 0.003)
        for i in range(1500):
            out = net.M_forwardPropagation(input_train)
            net.M_backwardPropagation(input_train, T_train)
        v_out = net.M_forwardPropagation(input_validation)
        v_errs.append(mean_sq_err(v_out, T_validation))
    mean_err.append(np.mean(v_errs))
    std_err.append(np.std(v_errs))

plt.figure(nf)
nf += 1
x_pos = np.arange(len(num_of_hnodes))
#plt.subplots_adjust(right=0.80)
plt.bar(x_pos, mean_err, align='center', zorder=4, color='cornflowerblue')
plt.grid(zorder=0, axis='y')
plt.xticks(x_pos, num_of_hnodes)
plt.ylabel('MSE')
plt.xlabel('Number of nodes in hidden layer')
plt.title(f'Mean of MSE of MLPs on validation set n=100')
for i, v in enumerate(mean_err):
    plt.text(x_pos[i] - 0.35, v + 0.001, '{:0.4e}'.format(v))

plt.figure(nf)
nf += 1
x_pos = np.arange(len(num_of_hnodes))
#plt.subplots_adjust(right=0.80)
plt.bar(x_pos, std_err, align='center', zorder=4, color='cornflowerblue')
plt.grid(zorder=0, axis='y')
plt.xticks(x_pos, num_of_hnodes)
plt.ylabel('Std of MSE')
plt.xlabel('Number of nodes in hidden layer')
plt.title(f'Std of MSE of MLPs on validation set n=100')
for i, v in enumerate(std_err):
    plt.text(x_pos[i] - 0.35, v + 0.0001, '{:0.4e}'.format(v))

plt.show()



# %%