"""
This program simulates diffusion limited aggregation on an LxL grid.
Particles are initiated until the centre point is filled.
Author: Nico Grisouard, University of Toronto
Based on Paul J Kushner's DAL-eample.py
Edited by: Sam De Abreu
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def nextmove(x, y):
    """ randomly choose a direction
    0 = up, 1 = down, 2 = left, 3 = right"""
    direction = np.random.randint(0, 4) 

    if direction == 0:  # move up
        y += 1
    elif direction == 1:  # move down
        y -= 1
    elif direction == 2:  # move right
        x += 1
    elif direction == 3:  # move left
        x -= 1
    else:
        print("error: direction isn't 0-3")
    return x, y


font = {'family': 'DejaVu Sans', 'size': 14}  # adjust fonts
rc('font', **font)

# %% main program starts here ------------------------------------------------|

Lp = 101  # size of domain
N = 100  # number of particles

anchored = np.zeros((Lp+1, Lp+1), dtype=int) # array to represent whether each gridpoint has an anchored particle

centre_point = (Lp-1)//2  # middle point of domain

# DLA algorithm
while anchored[centre_point, centre_point] != 1: # While there is no anchored particle at the center
    particle_position = (centre_point, centre_point)
    while 0 < particle_position[0] < Lp and 0 < particle_position[1] < Lp: # Check if particle is at the edges
        future_position = nextmove(*particle_position)
        if anchored[future_position[0], future_position[1]] == 1: # Check if the future position of particle is already anchored
            break 
        particle_position = future_position
    anchored[particle_position[0], particle_position[1]] = 1 # Anchor current particle and then start new at the center again

# Plotting
plt.imshow(anchored, origin='lower', extent=(0, 1, 0, 1))
plt.xlabel('$x/L$')
plt.ylabel('$y/L$')
plt.title('Final Position of all Anchored Particles')
plt.savefig('Q1b.png', dpi=300, bbox_inches='tight')
plt.clf()
    


