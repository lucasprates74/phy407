"""
This program simulates Brownian motion in the presence of walls
Note that the physical behaviour would be to stick to walls,
which is the purpose of Q1a.
Original author: Nico Grisouard, University of Toronto
Edited by: Sam De Abreu
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def nextmove(x, y):
    """ randomly choose a direction
    0 = up, 1 = down, 2 = left, 3 = right"""
    direction = np.random.randint(0, 4) 

    if direction == 0:  # move up
        if y < Lp: # If outside above
            y += 1
        else:
            return nextmove(x, y)
    elif direction == 1:  # move down
        if y > 0: # If outside below
            y -= 1
        else:
            return nextmove(x, y)
    elif direction == 2:  # move right
        if x < Lp: # If outside on the right side
            x += 1
        else:
            return nextmove(x, y)
    elif direction == 3:  # move left
        if x > 0: # If outside on the left side
            x -= 1
        else:
            return nextmove(x, y)
    else:
        print("error: direction isn't 0-3")
    return x, y


font = {'family': 'DejaVu Sans', 'size': 14}  # adjust fonts
rc('font', **font)


# %% main program starts here ------------------------------------------------|
# YOU NEED TO FINISH IT!


Lp = 101  # size of domain
Nt = 50000  # number of time steps
# arrays to record the trajectory of the particle
x_arr = np.zeros(Nt)
y_arr = np.zeros(Nt)

centre_point = (Lp-1)//2  # middle point of domain
xp = centre_point
yp = centre_point

# Position arrays
x_arr[0] = xp
y_arr[0] = yp

# Random walk loop
for i in range(Nt-1):
    x_arr[i+1], y_arr[i+1] = nextmove(x_arr[i], y_arr[i])

# Plotting
t = np.arange(0, Nt, 1)
fig, axs = plt.subplots(2, 1)
axs[0].plot(t, x_arr)
axs[0].grid()
axs[0].set_ylabel('$x(t_i)$')
axs[1].plot(t, y_arr)
axs[1].grid()
axs[1].set_ylabel('$y(t_i)$')
axs[1].set_xlabel('Time step $t_i$')
plt.suptitle('Time Series of $x(t_i)$ and $y(t_i)$')
plt.tight_layout()
plt.savefig('Q1aTimeSeries.png', dpi=300, bbox_inches='tight')
plt.clf()

plt.plot(x_arr, y_arr)
plt.grid()
plt.xlabel('$x(t_i)$')
plt.ylabel('$y(t_i)$')
plt.title('Particle Trajectory')
plt.savefig('Q1aTrajectory.png', dpi=300, bbox_inches='tight')
plt.clf()