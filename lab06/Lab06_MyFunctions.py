import numpy as np

DIMENSIONS = 2

def force(r1, r2): 
    # Return the force on particle 1 at r1 due to particle 2 at r2
    x1, y1 = r1
    x2, y2 = r2

    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    factor = (24 / dist ** 14) * (2 - dist ** 6)
    fx = (x1-x2) * factor
    fy = (y1-y2) * factor
    return np.array([fx, fy])

def tot_force(r, i):
    """
    return the force on particle i due to all other particles, assuming a 2D problem
    """
    r1 = r[i:i+2]
    tot = np.zeros(2)
    for j in range(0, len(r), DIMENSIONS):
        if j != i:
            r2 = r[j:j+2]
            tot += force(r1, r2)
    return tot

def solve(r0, v0, tstop, dt=0.3):
    DOFs = len(r0)  # get the number of degrees of freedom
    num_steps = int(tstop // dt)

    r = np.zeros((DOFs, num_steps))
    v = np.zeros((DOFs, num_steps))
    
    r[:, 0]=r0
    v[:, 0]=v0
    vhalf = np.zeros((DOFs, num_steps))
    for n in range(0, DOFs, DIMENSIONS):
        start, end = n, n+DIMENSIONS
        vhalf[start:end, 0] = v[start:end, 0] + ( dt / 2 ) * tot_force(r[:, 0], n)
    
    for i in range(1, num_steps):
        # compute the new positions for each particle
         for n in range(0, DOFs, DIMENSIONS):
            start, end = n, n + DIMENSIONS
            r[start:end, i] = r[start:end, i-1] + dt * vhalf[start:end, i-1]

        # compute the new velocities and half velocities
         for n in range(0, DOFs, DIMENSIONS):
            start, end = n, n + DIMENSIONS
            k = dt * tot_force(r[:, i], n)

            v[start:end, i] = vhalf[start:end, i-1] + 0.5 * k
            print(v[start:end, i])
            vhalf[start:end, i] = vhalf[start:end, i-1] + k
    return r, v


r0 = (4, 4, 5.2, 4)
v0 = (0,0,0,0)

r, v = solve(r0,v0,1)
x1, y1, x2, y2 = r

import matplotlib.pyplot as plt
plt.plot(x1, y1, marker='.', linestyle='none')
plt.plot(x2, y2, marker='.', linestyle='none')
plt.show()