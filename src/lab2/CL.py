from RBFNet import *
import matplotlib.pyplot as plt


def initialize_nodes_1(inputs: np.array, num_nodes: int,
                       avoid_dead_units: bool = False,
                       learning_rate: float = 0.1,
                       plot: bool = False):
    rbf_nodes = np.random.uniform(low=-3, high=6, size=num_nodes)
    if avoid_dead_units:
        rbf_nodes = inputs[np.random.choice(inputs.shape[0], num_nodes, replace=False)]

    sc = 0
    if plot:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.canvas.draw()
        plt.pause(1)
        plt.scatter(inputs, np.zeros(len(inputs)))
        sc = plt.scatter(rbf_nodes, np.zeros(len(rbf_nodes)), s=100)

    count = 0
    while True:
        for i in inputs:
            index = np.argmin(np.abs(rbf_nodes - i))
            delta = learning_rate * (i - rbf_nodes[index])
            rbf_nodes[index] += delta

        if count > 50:
            break

        if plot and count % 10 == 0:
            sc.set_offsets(np.c_[rbf_nodes, np.zeros(len(rbf_nodes))])
            plt.pause(1)

        count += 1

    if plot:
        plt.ioff()
        plt.show()

    return rbf_nodes


def initialize_nodes_2(inputs: np.array, num_nodes: int,
                       learning_rate: float = 0.1,
                       plot: bool = False):
    rbf_nodes = inputs[np.random.choice(inputs.shape[0], num_nodes, replace=False), :]

    sc = 0
    if plot:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fig.canvas.draw()
        plt.pause(1)
        plt.scatter(inputs[:, 0], inputs[:, 1])
        sc = plt.scatter(rbf_nodes[:, 0], rbf_nodes[:, 1], s=100)

    count = 0
    while True:
        for i in inputs:
            index = np.argmin(np.sum((rbf_nodes - i)**2, axis=-1)**(1./2))
            delta = learning_rate * (i - rbf_nodes[index])
            rbf_nodes[index] += delta

        if count > 50:
            break

        if plot and count % 10 == 0:
            sc.set_offsets(np.c_[rbf_nodes[:, 0], rbf_nodes[:, 1]])
            plt.pause(1)

        count += 1

    if plot:
        plt.ioff()
        plt.show()

    return rbf_nodes


def read_data(filename: str):
    f = open(filename, "r")
    input_data = []
    output_data = []
    for line in f.readlines():
        input_data.append(np.array(line.split()[:2], dtype=float))
        output_data.append(np.array(line.split()[2:], dtype=float))
    return np.vstack(input_data), np.vstack(output_data)


def task_1():
    train_inputs = np.arange(0, 2 * np.pi, 0.1)
    initialize_nodes_1(train_inputs, 20, True, 0.1, True)


def task_2():
    train_input_data, train_output_data = read_data("../Data/ballist.dat")
    test_input_data, test_output_data = read_data("../Data/balltest.dat")

    learning_rate = 0.1
    sigma = 0.1
    num_nodes = 30

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d', title="Distance predictions")
    ax2 = fig.add_subplot(1, 3, 2, projection='3d', title="Height predictions")
    ax3 = fig.add_subplot(1, 3, 3, title="Rbf nodes positions")

    xdata = test_input_data[:, 0]
    ydata = test_input_data[:, 1]
    ax1.scatter3D(xdata, ydata, test_output_data[:, 0], label="Generated")
    ax2.scatter3D(xdata, ydata, test_output_data[:, 1], label="Generated")

    nodes = initialize_nodes_2(train_input_data, num_nodes, learning_rate, False)

    ax3.scatter(train_input_data[:, 0], train_input_data[:, 1], label="Input data")
    ax3.scatter(nodes[:, 0], nodes[:, 1], s=100, label="Nodes")

    rbf1 = RBF(input_dim=2, sigma=sigma)
    rbf1.set_nodes(nodes)
    rbf_net1 = RBFNet(rbf1, num_of_outputs=2)
    rbf_net1.train(train_input_data, train_output_data)
    perr = rbf_net1.forward(test_input_data)
    ax1.scatter3D(xdata, ydata, perr[:, 0], label="Prediction")
    ax2.scatter3D(xdata, ydata, perr[:, 1], label="Prediction")

    rbf2 = RBF(input_dim=2, sigma=sigma)
    rbf2.set_nodes(nodes)
    rbf_net2 = RBFNet(rbf2, num_of_outputs=2)
    nepochs, _ = rbf_net2.train_online(train_input_data, train_output_data, learning_rate)
    perr = rbf_net2.forward(test_input_data)

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()


task_2()
