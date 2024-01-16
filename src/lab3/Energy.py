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

class PlotEnergy:
    def __init__(self):
        self.energies = []

    def update(self, p, cnt, subject):
        self.energies.append(subject.energy(p))
    
    def draw_plot(self, title):
        global nf
        plt.figure(nf)
        nf += 1
        plt.plot(self.energies, label='Energy')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title(title)
        plt.legend()
        

patterns = []
with open('pict.dat', 'r') as f:
    d = f.readline().strip().split(',')
    d = [int(x) for x in d]
    for i in range(11):
        patterns.append(d[:1024])
        d = d[1024:]
patterns = np.array(patterns)

# Energy of the attractors p1, p2, p3'
print('\nEnergy of the attractors p1, p2, p3!')
ahop = AsyncHopfield(patterns[:3,:], scale_weights=True)
for i in range(3):
    print(f'Energy at p{i+1}: {ahop.energy(np.reshape(patterns[i], (1,-1)))}')

print('\nEnergy of the distorted patterns p10 and p11!')
print(f'Energy at p{10}: {ahop.energy(np.reshape(patterns[9], (1,-1)))}')
print(f'Energy at p{11}: {ahop.energy(np.reshape(patterns[10], (1,-1)))}')

print('\nTracking energy during recall for pattern p10')
en_listener = PlotEnergy()
ahop.update_listener = en_listener
new_x, num = ahop.update_until_converge(np.reshape(patterns[9], (1,-1)))
en_listener.draw_plot('Energy of the Hopfield network during recall for p10')
print(f'Final energy: {en_listener.energies[-1]}')


print('\nWhat happens when we set the weights to random normally distributed numbers?')
ahop.W = np.random.normal(size=ahop.W.shape)
en_listener = PlotEnergy()
ahop.update_listener = en_listener
ahop.limit = 30
new_x, num = ahop.update_until_converge(np.reshape(patterns[9], (1,-1)))
en_listener.draw_plot('Energy of the Hopfield network during recall for p10\nwith a random weight matrix')
print(f'Final energy: {en_listener.energies[-1]}')
draw_img(new_x, 'Output for p10 of Hopfield net with random weights')

print('\nAfter making the random weight matrix symmetric, this is what happens:')
ahop.W = 0.5*(ahop.W+ahop.W.T)
en_listener = PlotEnergy()
ahop.update_listener = en_listener
ahop.limit = 50
print('Limit is 51200 iterations')
new_x, num = ahop.update_until_converge(np.reshape(patterns[9], (1,-1)))
en_listener.draw_plot('Energy of the Hopfield network during recall for p10\nwith a random but symmetric weight matrix')
print(f'Final energy: {en_listener.energies[-1]}')
draw_img(new_x, 'Output for p10 of Hopfield net with random, but symmetric weight matrix')
print(f'Is the output of the network stable: {ahop.is_attractor(new_x)}')

plt.show()

