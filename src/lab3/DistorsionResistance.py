from LittleModel import LittleModel
from AsyncHopfield import AsyncHopfield
import numpy as np


def test_tolerance(lm: LittleModel, am: AsyncHopfield, pattern: np.array, noise_amount: float):
    selected_indices = np.random.choice(pattern.shape[0], int(noise_amount * len(pattern)), replace=False)
    mutated_pattern = pattern.copy()
    for i in selected_indices:
        if mutated_pattern[i] == 1:
            mutated_pattern[i] = -1
        else:
            mutated_pattern[i] = 1

    new_x_s, num_s = lm.update_until_converge(np.reshape(mutated_pattern, (1, -1)))
    new_x_a, num_a = am.update_until_converge(np.reshape(mutated_pattern, (1, -1)))

    return np.array_equal(new_x_s[0], pattern), np.array_equal(new_x_a[0], pattern)


def test_pattern(lm: LittleModel, am: AsyncHopfield, pattern: np.array):
    i_sum_s = 0
    i_sum_a = 0
    iteration_count = 10
    for j in range(iteration_count):
        i_add_s = 101
        i_add_a = 101
        for i in range(100):
            s, a = test_tolerance(lm, am, pattern, i / 100.0)
            if not s:
                i_add_s = min(i - 1, i_add_s)
            if not a:
                i_add_a = min(i - 1, i_add_a)
            if not s and not a:
                break
        i_sum_s += i_add_s
        i_sum_a += i_add_a

    print(f"avg tolerance sync = {float(i_sum_s) / iteration_count}%")
    print(f"avg tolerance async = {float(i_sum_a) / iteration_count}%")


patterns = []
with open('pict.dat', 'r') as f:
    d = f.readline().strip().split(',')
    d = [int(x) for x in d]
    for i in range(11):
        patterns.append(d[:1024])
        d = d[1024:]
patterns = np.array(patterns)

lm = LittleModel(patterns[:3, :])
am = AsyncHopfield(patterns[:3, :])

test_pattern(lm, am, patterns[0])
test_pattern(lm, am, patterns[1])
test_pattern(lm, am, patterns[2])
