import numpy as np
from RBF import *
from RBFNet import *
import matplotlib.pyplot as plt

def f1(x):   # sin(2x)
    return np.sin(2*x)

def f2(x):   # envelope for sin(2x)
    if f1(x) >= 0: return 1
    else: return -1

def step_func(x):
    if x >= 0: return 1
    else: return -1

def total_error(out, T): # only for 1d outouts
    return distance(out, T)**2

nf = 1
f2 = np.vectorize(f2)
step_func = np.vectorize(step_func)
x_pos = np.arange(0, 2*np.pi, 0.1)
x_pos_validation = np.arange(0.05, 2*np.pi, 0.1)
f1_values = f1(x_pos)
f2_values = f2(x_pos)
f1_values_validation = f1(x_pos_validation)
f2_values_validation = f2(x_pos_validation)

#data
training_patterns_sin = x_pos.reshape((-1,1))
training_tragets_sin = f1_values.reshape((-1,1))

training_patterns_envelope = x_pos.reshape((-1,1))
training_tragets_envelope = f2_values.reshape((-1,1))

validation_patterns_sin = x_pos_validation.reshape((-1,1))
validation_tragets_sin = f1_values_validation.reshape((-1,1))

validation_patterns_envelope = x_pos_validation.reshape((-1,1))
validation_tragets_envelope = f2_values_validation.reshape((-1,1))


num_of_nodes=10
sigma=1.5
node_pos1 = np.linspace(0, 2*np.pi, num_of_nodes)
node_pos1 = node_pos1.reshape((-1,1))
rbf1 = RBF(input_dim=1, sigma=sigma)
rbf1.set_nodes(node_pos1)
rbf_net = RBFNet(rbf1, num_of_outputs=1)

# rbf_net.W *= 100
print(rbf_net.W)
perr = rbf_net.forward(validation_patterns_sin)
aerr = absolute_residual_error(perr, validation_tragets_sin)
print('ONLINE')
print(f'err bf training: {aerr}')

nepochs, err = rbf_net.train_online(training_patterns_sin, training_tragets_sin, 0.01)
print(f'ne: {nepochs}, err: {err}')
print(rbf_net.W)
print(f'num of nodes: {num_of_nodes}, sigma: {sigma}')

plt.figure(1)
plt.plot(x_pos, f1_values, label='sin(2x)')
pred = rbf_net.forward(training_patterns_sin).T[0]
plt.plot(x_pos, pred, label='sin(2x) prediction')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Predictions for model with online learning\nnodes={num_of_nodes}, sigma={sigma}, epochs={nepochs}, err={"{:.3f}".format(err)}')
plt.show()



# perr = rbf_net.forward(validation_patterns_sin)
# aerr = absolute_residual_error(perr, validation_tragets_sin)
# print(f'err bf training: {aerr}')

# rbf_net.train(training_patterns_sin, training_tragets_sin)

# perr = rbf_net.forward(validation_patterns_sin)
# aerr = absolute_residual_error(perr, validation_tragets_sin)
# print(f'err af training: {aerr}')
# print(rbf_net.W)
# print(f'num of nodes: {num_of_nodes}, sigma: {sigma}')
