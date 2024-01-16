import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim, from_numpy


def precompute_series(num_iter):
    _series = np.empty(num_iter + 1, dtype=float)
    _series[0] = 1.5

    for i in range(num_iter):
        past = 0
        if i >= 25:
            past = _series[i - 25]
        value = _series[i] + (0.2 * past) / (1 + pow(past, 10)) - 0.1 * _series[i]
        _series[i + 1] = value

    return _series


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers: list):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        last_size = 5
        layers = []
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(last_size, hidden_layers[i]))
            layers.append(nn.Sigmoid())
            last_size = hidden_layers[i]
        layers.append(nn.Linear(last_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        self.flatten(x)
        return self.network(x)


def main():
    series = precompute_series(1600)

    train_size = 1000
    validation_size = 100
    num_epochs = 100000
    hidden_layers = [5, 10]

    model = NeuralNetwork(hidden_layers)

    t_list = range(301, 1500)
    input_gen = np.empty((len(t_list), 5), dtype=float)
    output_gen = np.empty((len(t_list), 1), dtype=float)
    for i, t in enumerate(t_list):
        for j in range(5):
            input_gen[i][j] = series[t - (4 - j) * 5]
        output_gen[i][0] = series[t + 5]

    input_train = from_numpy(input_gen[:train_size]).float()
    output_train = from_numpy(output_gen[:train_size]).float()

    input_val = from_numpy(input_gen[train_size:validation_size + train_size]).float()
    output_val = from_numpy(output_gen[train_size:validation_size + train_size]).float()

    input_test = from_numpy(input_gen[train_size:]).float()

    output_gen = from_numpy(output_gen).float()

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    epochs = range(num_epochs)
    loss_list = []
    loss_val_list = []
    for epoch in range(num_epochs):
        print(f'e: {epoch}')
        output_pred = model(input_train)
        loss = criterion(output_pred, output_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach().numpy() / train_size)

        output_val_pred = model(input_val)
        loss_val = criterion(output_val_pred, output_val)
        loss_val_list.append(loss_val.detach().numpy() / validation_size)

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(epochs, loss_list, label="Train loss")
    ax1.plot(epochs, loss_val_list, label="Validation loss")
    ax1.legend()

    t_train = range(301, 301 + train_size)
    t_test = range(301 + train_size, 1500)
    output_pred_train = model(input_train)
    output_pred_test = model(input_test)
    ax2.plot(t_train, output_pred_train.detach().numpy(), label="Predicted - train set")
    ax2.plot(t_test, output_pred_test.detach().numpy(), label="Predicted - validation + test set")
    ax2.plot(t_list, output_gen.detach().numpy(), label="Generated")
    ax2.legend()

    plt.show()


main()
