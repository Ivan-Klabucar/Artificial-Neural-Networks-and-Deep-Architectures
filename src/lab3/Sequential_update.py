import numpy as np
import matplotlib.pyplot as plt 
from LittleModel import LittleModel
from AsyncHopfield import AsyncHopfield

global nf
nf = 1

def draw_img(p, title):
    global nf
    for_drawing = np.reshape(p, (32,32)).T
    for_drawing = 1 + for_drawing
    for_drawing = 0.5 * for_drawing
    plt.figure(nf)
    plt.imshow(for_drawing, cmap="gray") 
    plt.title(title)
    nf+=1

class DrawEveryOneHundred:
    def __init__(self, title):
        self.title = title

    def update(self, p, cnt, subject):
        if cnt > 1400: return
        if cnt % 100 == 0:
            draw_img(p, f'{self.title}{cnt}')

patterns = []
with open('pict.dat', 'r') as f:
    d = f.readline().strip().split(',')
    d = [int(x) for x in d]
    for i in range(11):
        patterns.append(d[:1024])
        d = d[1024:]
patterns = np.array(patterns)

# for p in patterns:
#     draw_img(p, f'Picture #{nf}')

# 'Testing The Little model on the images!'
print('\nTesting The Little model on the images!')
print('We trained a Little Model on first three patterns')
print('Let us check if all three patterns are stable')
lm = LittleModel(patterns[:3,:])
for i in range(3):
    if lm.is_attractor(np.reshape(patterns[i], (1,-1))): print(f'p{i+1} is stable.')
    else: print(f'p{i+1} is unstable.')

print('\nLet\'s test the network on p10')
print('which is a degraded version of p1')
new_x, num = lm.update_until_converge(np.reshape(patterns[9], (1,-1)))
print(f'Converged in {num} steps!')
if (new_x[0] == patterns[0]).all(): print('The network successfully recovered p1 from p10!')
else: print('The network did not recover p1 from p10!')
draw_img(new_x, 'Little Model\'s output for p10')

print('\nLet\'s test the network on p11')
print('which is a mixture of p2 and p3')
new_x, num = lm.update_until_converge(np.reshape(patterns[10], (1,-1)))
print(f'Converged in {num} steps!')
if (new_x[0] == patterns[1]).all(): print('The network successfully recovered p2 from p11!')
elif (new_x[0] == patterns[2]).all(): print('The network successfully recovered p3 from p11!')
else: 
    print('The network did not recover p2 or p3 from p11!')
    print(f'Since it returned neither p2 or p3, is the output of the network stable: {lm.is_attractor(new_x)}')
draw_img(new_x, 'Little Model\'s output for p11')

# Testing the AsyncHopfield on the images!
print('\nTesting the AsyncHopfield on the images!')
print('We trained an AsyncHopfield on first three patterns')
print('Let us check if all three patterns are stable')
ahop = AsyncHopfield(patterns[:3,:])
for i in range(ahop.W.shape[0]):
    ahop.W[i][i] = 0
for i in range(3):
    if ahop.is_attractor(np.reshape(patterns[i], (1,-1))): print(f'p{i+1} is stable.')
    else: print(f'p{i+1} is unstable.')

print('\nLet\'s test the network on p10')
print('which is a degraded version of p1')
new_x, num = ahop.update_until_converge(np.reshape(patterns[9], (1,-1)))
print(f'Converged in {num} steps!')
if (new_x[0] == patterns[0]).all(): print('The network successfully recovered p1 from p10!')
else: print('The network did not recover p1 from p10!')
draw_img(new_x, 'AsyncHopfield\'s output for p10')

print('\nLet\'s test the network on p11')
print('which is a mixture of p2 and p3')
new_x, num = ahop.update_until_converge(np.reshape(patterns[10], (1,-1)))
print(f'Converged in {num} steps!')
if (new_x[0] == patterns[1]).all(): print('The network successfully recovered p2 from p11!')
elif (new_x[0] == patterns[2]).all(): print('The network successfully recovered p3 from p11!')
else: print('The network did not recover p2 or p3 from p11!')
draw_img(new_x, 'AsyncHopfield\'s output for p11')


# Capturing the state of the AsyncHopfield every 100 iterations for p10
print('\nCapturing the state of the AsyncHopfield every 100 iterations for p10')
listener = DrawEveryOneHundred('AsyncHopfield\'s output for p10 after update #')
ahop.update_listener = listener
new_x, num = ahop.update_until_converge(np.reshape(patterns[9], (1,-1)))





print('\nBlack: -1, White: 1')
plt.show()

