from RBF import *
from RBFNet import *
import numpy as np
import matplotlib.pyplot as plt

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

def train_online_and_batch(nodes, \
                           sg, \
                           training_patterns, \
                           training_tragets, \
                           validation_patterns, \
                           validation_tragets, \
                           eta):

    node_pos1 = np.linspace(0, 2*np.pi, nodes)
    node_pos1 = node_pos1.reshape((-1,1))
    
    rbf1 = RBF(input_dim=1, sigma=sg)
    rbf1.set_nodes(node_pos1)
    rbf_net1 = RBFNet(rbf1, num_of_outputs=1)
    rbf_net1.train(training_patterns, training_tragets)
    perr = rbf_net1.forward(validation_patterns)
    aerr_batch = absolute_residual_error(perr, validation_tragets)

    rbf2 = RBF(input_dim=1, sigma=sg)
    rbf2.set_nodes(node_pos1)
    rbf_net2 = RBFNet(rbf2, num_of_outputs=1)
    nepochs, _ = rbf_net2.train_online(training_patterns, training_tragets, eta)
    perr = rbf_net2.forward(validation_patterns)
    aerr_online = absolute_residual_error(perr, validation_tragets)
    return aerr_batch, aerr_online, nepochs

def check_random(training_patterns, training_tragets, validation_patterns, validation_tragets):
    global nf
    rand = []
    lin = []
    nodes = 26
    sg = 0.5
    
    for i in range(100):    
        node_pos = np.linspace(0, 2*np.pi, nodes)
        node_pos = node_pos.reshape((-1,1))

        node_pos_rand = np.random.random((nodes)) * 2*np.pi
        node_pos_rand = node_pos_rand.reshape((-1,1))

        rbf1 = RBF(input_dim=1, sigma=sg)
        rbf1.set_nodes(node_pos)
        rbf_net1 = RBFNet(rbf1, num_of_outputs=1)
        rbf_net1.train(training_patterns, training_tragets)
        perr = rbf_net1.forward(validation_patterns)
        aerr_lin = absolute_residual_error(perr, validation_tragets)
        lin.append(aerr_lin)

        rbf2 = RBF(input_dim=1, sigma=sg)
        rbf2.set_nodes(node_pos_rand)
        rbf_net2 = RBFNet(rbf2, num_of_outputs=1)
        rbf_net2.train(training_patterns, training_tragets)
        perr = rbf_net2.forward(validation_patterns)
        aerr_rand = absolute_residual_error(perr, validation_tragets)
        rand.append(aerr_rand)
    
    fig = plt.figure(nf)
    nf += 1
 
    # Creating axes instance
   #ax = fig.add_axes([0, 0, 1, 1])
    
    data = [lin, rand]
    bp = plt.boxplot(data, labels=['linearly spaced', 'randomly spaced'])
    print(f'meanlin: {np.mean(lin)}, mean rand: {np.mean(rand)}')

    
    plt.title('Absolute residual error of models with architecture nodes=26, sigma=0.5\nn=100')



def grid_experiment(training_patterns, \
                    training_tragets, \
                    validation_patterns, \
                    validation_tragets, \
                    eta, \
                    title):
    global nf
    sigma = [0.1, 0.25, 0.5, 1, 1.5]
    num_of_nodes = [10, 15, 20, 30, 60]
    colors_row = plt.cm.BuPu(np.linspace(0, 0.5, len(num_of_nodes)))
    colors_col = plt.cm.Oranges(np.linspace(0, 0.5, len(num_of_nodes)))

    cell_text = []

    for nodes in num_of_nodes:
        cell_row = []
        for sg in sigma:
            aerr_batch, aerr_online, nepochs = train_online_and_batch(nodes, sg, training_patterns, training_tragets, validation_patterns, validation_tragets, eta)
            cell_row.append('batch_err={:.4f}\nonline_err={:.4f}\nonline_epochs={}'.format(aerr_batch, aerr_online, nepochs))
            print(f'{title}, n:{nodes}, s:{sg}, batch_err={aerr_batch}\nonline_err={aerr_online}\nonline_epochs={nepochs}')
        cell_text.append(cell_row)
        print(f'{title}, done with nodes={nodes}')
    
    plt.figure(nf)
    nf += 1
    plt.box(on=None)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    the_table = plt.table(cellText=cell_text,
                        rowLabels=num_of_nodes,
                        rowColours=colors_row,
                        colColours=colors_col,
                        colLabels=sigma, loc='center')
    the_table.scale(1, 3.5)
    plt.title(title)
    


f2 = np.vectorize(f2)
step_func = np.vectorize(step_func)
x_pos = np.arange(0, 2*np.pi, 0.1)
x_pos_validation = np.arange(0.05, 2*np.pi, 0.1)
f1_values = f1(x_pos)
f2_values = f2(x_pos)
f1_values_validation = f1(x_pos_validation)
f2_values_validation = f2(x_pos_validation)

#non noisy data
training_patterns_sin = x_pos.reshape((-1,1))
training_tragets_sin = f1_values.reshape((-1,1))

training_patterns_envelope = x_pos.reshape((-1,1))
training_tragets_envelope = f2_values.reshape((-1,1))

validation_patterns_sin = x_pos_validation.reshape((-1,1))
validation_tragets_sin = f1_values_validation.reshape((-1,1))

validation_patterns_envelope = x_pos_validation.reshape((-1,1))
validation_tragets_envelope = f2_values_validation.reshape((-1,1))

#noisy data
noise_mean = 0
noise_std = 0.07
plt.figure(nf)
nf += 1


noisy_training_tragets_sin = training_tragets_sin + np.random.normal(noise_mean, noise_std, training_tragets_sin.shape)
noisy_training_tragets_envelope = training_tragets_envelope + np.random.normal(noise_mean, noise_std, training_tragets_envelope.shape)

plt.plot(training_patterns_sin, noisy_training_tragets_sin, label='noisy sin(2x)')
plt.plot(training_patterns_envelope, noisy_training_tragets_envelope, label='noisy square(2x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower left')
plt.title('Noisy dataset used for training')


# grid_experiment(training_patterns_sin, training_tragets_sin, validation_patterns_sin, validation_tragets_sin, 0.01, 'Performance of online and batch models on non noisy data sin(2x)')
# grid_experiment(training_patterns_envelope, training_tragets_envelope, validation_patterns_envelope, validation_tragets_envelope, 0.01, 'Performance of online and batch models on non noisy data square(2x)')
# grid_experiment(training_patterns_sin, noisy_training_tragets_sin, validation_patterns_sin, validation_tragets_sin, 0.01, 'Performance of online and batch models on noisy data sin(2x)')
# grid_experiment(training_patterns_envelope, noisy_training_tragets_envelope, validation_patterns_envelope, validation_tragets_envelope, 0.01, 'Performance of online and batch models on noisy data square(2x)')


# plt.figure(nf)
# nf+=1
# plt.plot([1,2,3], [4,5,6], label='number of nodes', color='blue')
# plt.plot([1,2,3], [5,6,7], label='sigma', color='orange')
# plt.legend(fontsize=15, handlelength=0.7)

#check_random(training_patterns_sin, training_tragets_sin, validation_patterns_sin, validation_tragets_sin)

plt.show()




