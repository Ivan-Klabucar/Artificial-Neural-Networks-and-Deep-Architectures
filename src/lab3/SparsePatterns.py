from BiasModel import BiasModel
import numpy as np
import matplotlib.pyplot as plt


def test_sparsity(pattern_size: int, bias: float, active_amount: float):
    i_sum = 0
    for j in range(1):
        i_sum += 100
        patterns = []
        for i in range(100):
            new_pattern = np.zeros(pattern_size)
            selected_indices = np.random.choice(pattern_size, int(active_amount * pattern_size), replace=False)
            for index in selected_indices:
                new_pattern[index] = 1
            patterns.append(new_pattern)
            bm = BiasModel(np.array(patterns))
            success = True
            for pattern in patterns:
                if not bm.is_attractor(pattern, bias):
                    success = False
                    break

            if not success:
                i_sum += i - 100
                break

    return i_sum / 1.0


fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 3, 1, title="Sparsity = 0.1")
ax2 = fig.add_subplot(1, 3, 2, title="Sparsity = 0.05")
ax3 = fig.add_subplot(1, 3, 3, title="Sparsity = 0.01")

biases = np.arange(0, 1, 0.1)

values = []
for b in biases:
    values.append(test_sparsity(30, b, 0.1))
ax1.plot(biases, values)

values = []
for b in biases:
    values.append(test_sparsity(30, b, 0.05))
ax2.plot(biases, values)

values = []
for b in biases:
    values.append(test_sparsity(30, b, 0.01))
ax3.plot(biases, values)

ax1.set_xlabel("Bias")
ax1.set_ylabel("Max number of stable patterns")
ax2.set_xlabel("Bias")
ax2.set_ylabel("Max number of stable patterns")
ax3.set_xlabel("Bias")
ax3.set_ylabel("Max number of stable patterns")
plt.show()
