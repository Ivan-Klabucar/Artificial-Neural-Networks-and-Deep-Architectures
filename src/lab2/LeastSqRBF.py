from RBF import *
from RBFNet import *
import numpy as np
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

def absolute_residual_error(out, T):
    return np.mean(np.absolute(out - T))

nf = 1
f2 = np.vectorize(f2)
step_func = np.vectorize(step_func)
x_pos = np.arange(0, 2*np.pi, 0.1)
x_pos_validation = np.arange(0.05, 2*np.pi, 0.1)
f1_values = f1(x_pos)
f2_values = f2(x_pos)
f1_values_validation = f1(x_pos_validation)
f2_values_validation = f2(x_pos_validation)
plt.figure(nf)
nf += 1
plt.plot(x_pos, f1_values, label='sin(2x)')
plt.plot(x_pos, f2_values, label='square(2x)')


#data
training_patterns_sin = x_pos.reshape((-1,1))
training_tragets_sin = f1_values.reshape((-1,1))

training_patterns_envelope = x_pos.reshape((-1,1))
training_tragets_envelope = f2_values.reshape((-1,1))

validation_patterns_sin = x_pos_validation.reshape((-1,1))
validation_tragets_sin = f1_values_validation.reshape((-1,1))

validation_patterns_envelope = x_pos_validation.reshape((-1,1))
validation_tragets_envelope = f2_values_validation.reshape((-1,1))


node_pos1 = np.linspace(0, 2*np.pi, 25)
node_pos1 = node_pos1.reshape((-1,1))
rbf1 = RBF(input_dim=1, sigma=0.25)
rbf1.set_nodes(node_pos1)
rbf_net = RBFNet(rbf1, num_of_outputs=1)
rbf_net.train(training_patterns_sin, training_tragets_sin)
pred = rbf_net.forward(training_patterns_sin).T[0]
perr = rbf_net.forward(validation_patterns_sin)
aerr = absolute_residual_error(perr, validation_tragets_sin)
print(f'aerr: {aerr}')
plt.plot(x_pos, pred, label='sin(2x) prediction')
plt.legend()
# plt.show()
# quit()

plt.figure(nf)
nf += 1

# plt.plot(x_pos_validation, validation_tragets_envelope-0.07, label='targets')
# plt.plot(x_pos, training_tragets_envelope, label='training')
# plt.legend()

node_num = 800
node_pos2 = np.linspace(0, 2*np.pi, node_num)
node_pos2 = node_pos2.reshape((-1,1))
sg=0.038
rbf1 = RBF(input_dim=1, sigma=sg)
rbf1.set_nodes(node_pos2)
rbf_net = RBFNet(rbf1, num_of_outputs=1)
rbf_net.train(training_patterns_envelope, training_tragets_envelope)
pred = rbf_net.forward(validation_patterns_envelope)
pred_for_plot = pred.T[0]
plt.plot(x_pos_validation, pred_for_plot, label='prediction')
plt.plot(x_pos_validation, step_func(pred_for_plot), label='step(prediction)')
plt.plot(x_pos, f1_values, label='sin(2x)')
plt.axhline(y=0.0)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Predictions of the envelope func, rbf nodes={node_num}, sigma={sg}')
print(f'abs. residual err without step transform, sg={sg}, n={node_num}: {absolute_residual_error(pred, validation_tragets_envelope)}')
#print(f'abs. residual err: {absolute_residual_error(step_func(pred), validation_tragets_envelope)}')
plt.legend(loc='lower left')
plt.show()
quit()


# l = [ 0.1, 0.01,0.001]
# results = dict()
# num_of_nodes = 3
# sg = 1.5
# while l:
#     node_pos1 = np.linspace(0, 2*np.pi, num_of_nodes)
#     node_pos1 = node_pos1.reshape((-1,1))
    
#     rbf1 = RBF(input_dim=1, sigma=sg)
#     rbf1.set_nodes(node_pos1)
#     rbf_net = RBFNet(rbf1, num_of_outputs=1)
#     rbf_net.train(training_patterns_sin, training_tragets_sin)
#     perr = rbf_net.forward(validation_patterns_sin)
#     aerr = absolute_residual_error(perr, validation_tragets_sin)
#     print(f'({aerr},{num_of_nodes},)')
#     if aerr < l[0]:
#         results[l[0]] = {'err': aerr, 'num_of_nodes': num_of_nodes}
#         l.pop(0)
#         print('YAS')
#     num_of_nodes+=1

# print(f'results for sin(2x), sg={sg}:')
# print(results)
# quit()
l = [ 0.1, 0.01,0.001]
results = dict()
num_of_nodes = 3
sg = 1
while l:
    node_pos1 = np.linspace(0, 2*np.pi, num_of_nodes)
    node_pos1 = node_pos1.reshape((-1,1))
    sg = (2*np.pi)/(2*num_of_nodes)
    if sg < 0.04: sg = 0.04
    rbf1 = RBF(input_dim=1, sigma=sg)
    rbf1.set_nodes(node_pos1)
    rbf_net = RBFNet(rbf1, num_of_outputs=1)
    rbf_net.train(training_patterns_envelope, training_tragets_envelope)
    perr = rbf_net.forward(validation_patterns_envelope)
    aerr = absolute_residual_error(perr, validation_tragets_envelope)
    print(f'({aerr},{num_of_nodes},{sg})')
    if aerr < l[0]:
        results[l[0]] = {'err': aerr, 'num_of_nodes': num_of_nodes}
        l.pop(0)
        print('YAS')
    num_of_nodes+=1

print(f'results for envelope, sg={sg}:')
print(results)
quit()


