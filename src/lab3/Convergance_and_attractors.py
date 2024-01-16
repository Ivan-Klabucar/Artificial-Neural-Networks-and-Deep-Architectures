import numpy as np
from LittleModel import LittleModel

def int_to_pattern(i):
    s = '{0:08b}'.format(i)
    mapping = {'0': -1, '1': 1}
    l = [mapping[c] for c in s]
    return np.reshape(np.array(l), (1,-1))

X = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],   # x1
              [-1, -1, -1, -1, -1, 1, -1, -1], # x2
              [-1, 1, 1, -1, -1, 1, -1, 1]])   # x3


print('Training matrix, every row is an input pattern:')
print(X)


x1d=np.array([[1, -1, 1, -1, 1, -1, -1, 1]])
x2d=np.array([[1, 1, -1, -1, -1, 1, -1, -1]])
x3d=np.array([[1, 1, 1, -1, 1, 1, -1, 1]])
noisy_patterns = [x1d, x2d, x3d]

very_noisy_x1d = np.array([[1, 1, -1, -1, -1, -1, -1, -1]])
very_noisy_x2d = np.array([[1, -1, 1, -1, -1, -1, 1, 1]])
very_noisy_x3d = np.array([[1, -1, -1, -1, 1, -1, -1, 1]])
very_noisy_patterns = [very_noisy_x1d, very_noisy_x2d, very_noisy_x3d]
print()

lm = LittleModel(X)
print(lm.W)
for i in range(lm.W.shape[0]):
    lm.W[i][i] = 0
print()
print(lm.W)

for i in range(3):
    new_x, num = lm.update_until_converge(np.reshape(X[i], (1,-1)))
    print(f'pattern {X[i]} took {num} steps to converge')

for i, npat in enumerate(noisy_patterns):
    new_x, num = lm.update_until_converge(npat)
    print(f'pattern {npat} took {num} steps to converge to {new_x}, success={(new_x == np.reshape(X[i], (1, -1))).all()}')

# Checking all patterns to see which are attractors
patterns_to_check = 2**8
print(f'\nFound attractors:')
for i in range(patterns_to_check):
    p = int_to_pattern(i)
    if lm.is_attractor(p): print(p)


print('\nVery noisy patterns:')
for i, npat in enumerate(very_noisy_patterns):
    new_x, num = lm.update_until_converge(npat)
    print(f'pattern {npat} took {num} steps to converge to {new_x}, success={(new_x == np.reshape(X[i], (1, -1))).all()}')





