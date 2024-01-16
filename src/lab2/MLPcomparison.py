import numpy as np
import matplotlib.pyplot as plt
from TwoLayerPerceptron import TwoLayerPerceptron
from RBFNet import absolute_residual_error

global nf
nf = 1

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



f2 = np.vectorize(f2)
step_func = np.vectorize(step_func)
x_pos = np.arange(0, 2*np.pi, 0.1)
x_pos_validation = np.arange(0.05, 2*np.pi, 0.1)
f1_values = f1(x_pos)
f2_values = f2(x_pos)
f1_values_validation = f1(x_pos_validation)
f2_values_validation = f2(x_pos_validation)


#non noisy data
training_patterns_sin = x_pos.reshape((1,-1))
training_tragets_sin = f1_values.reshape((1,-1))

training_patterns_envelope = x_pos.reshape((1,-1))
training_tragets_envelope = f2_values.reshape((1,-1))

validation_patterns_sin = x_pos_validation.reshape((1,-1))
validation_tragets_sin = f1_values_validation.reshape((1,-1))

validation_patterns_envelope = x_pos_validation.reshape((1,-1))
validation_tragets_envelope = f2_values_validation.reshape((1,-1))

# print(f's: {validation_patterns_sin.shape}')
# print(validation_patterns_sin)
# quit()

#noisy data
noise_mean = 0
noise_std = 0.07

noisy_training_tragets_sin = training_tragets_sin + np.random.normal(noise_mean, noise_std, training_tragets_sin.shape)
noisy_training_tragets_envelope = training_tragets_envelope + np.random.normal(noise_mean, noise_std, training_tragets_envelope.shape)

# NOISY SIN MLP
# n_h=10
# net = TwoLayerPerceptron(n_in=1, n_h=n_h, n_out=1, eta=0.004, alfa=0.7)
# err = []
# v_err = []
# ne =15000
# for i in range(ne):
#     v_out = net.M_forwardPropagation(validation_patterns_sin)
#     out = net.M_forwardPropagation(training_patterns_sin)
#     are_train = absolute_residual_error(out, noisy_training_tragets_sin)
#     v_are = absolute_residual_error(v_out, validation_tragets_sin)
#     err.append(are_train)
#     v_err.append(v_are)
#     net.M_backwardPropagation(training_patterns_sin, noisy_training_tragets_sin)
# fig = plt.figure(nf)
# nf+=1
# ax = plt.axes()
# ax.plot(range(ne), err, color='cornflowerblue', label='training error')
# print(f'verr: {v_err[-1]}')
# print(f'terr: {err[-1]}')
# ax.plot(range(ne), v_err, color='indianred', label='validation error')
# ax.legend()
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Absolute Residual Error')
# plt.title(f'MLP\'s Error during training on noisy sin(2x)\nn_h={n_h}')

# fig = plt.figure(nf)
# nf+=1
# plt.plot(validation_patterns_sin.reshape(-1,1), validation_tragets_sin.reshape(-1,1), label='validation target')
# plt.plot(training_patterns_sin.reshape(-1,1), noisy_training_tragets_sin.reshape(-1,1), label='noisy training')
# plt.plot(validation_patterns_sin.reshape(-1,1), net.M_forwardPropagation(validation_patterns_sin).reshape(-1,1), label='validation prediction')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'Predictions made by MLP on noisy sin(2x)\nn_h={n_h}')
# plt.legend(loc='lower left')

# NOISY SQUARE
n_h=30
net = TwoLayerPerceptron(n_in=1, n_h=n_h, n_out=1, eta=0.004, alfa=0.7)
err = []
v_err = []
ne =20000
for i in range(ne):
    v_out = net.M_forwardPropagation(validation_patterns_envelope)
    out = net.M_forwardPropagation(training_patterns_envelope)
    are_train = absolute_residual_error(out, noisy_training_tragets_envelope)
    v_are = absolute_residual_error(v_out, validation_tragets_envelope)
    err.append(are_train)
    v_err.append(v_are)
    net.M_backwardPropagation(training_patterns_envelope, noisy_training_tragets_envelope)
fig = plt.figure(nf)
nf+=1
ax = plt.axes()
ax.plot(range(ne), err, color='cornflowerblue', label='training error')
print(f'verr: {v_err[-1]}')
print(f'terr: {err[-1]}')
ax.plot(range(ne), v_err, color='indianred', label='validation error')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Absolute Residual Error')
plt.title(f'MLP\'s Error during training on noisy square(2x)\nn_h={n_h}')

fig = plt.figure(nf)
nf+=1
plt.plot(validation_patterns_envelope.reshape(-1,1), validation_tragets_envelope.reshape(-1,1), label='validation target')
plt.plot(training_patterns_envelope.reshape(-1,1), noisy_training_tragets_envelope.reshape(-1,1), label='noisy training')
plt.plot(validation_patterns_envelope.reshape(-1,1), net.M_forwardPropagation(validation_patterns_envelope).reshape(-1,1), label='validation prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Predictions made by MLP on noisy square(2x)\nn_h={n_h}')
plt.legend(loc='lower left')

plt.show()

