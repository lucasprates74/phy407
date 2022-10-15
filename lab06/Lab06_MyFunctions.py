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

def solve(r0, v0, tstop, dt=0.01):
    """
    Given initial conditions r0 = (x1, y1, ... , xN, yN) and v0 for an N
    particle system, solves the system of ODEs using Verlet Algorithm.
    """
    DOFs = len(r0)  # get the number of degrees of freedom
    num_steps = tstop // dt  # get the number of steps

    r = np.zeros((DOFs, num_steps))  # [coordinate][time]
    v = np.zeros((DOFs, num_steps))  # [coordinate][time]
    vhalf = np.zeros((DOFs, num_steps))  # half velocities for verlet algorithm
    
    # set initial conditions
    r[:, 0]=r0
    v[:, 0]=v0

    # compute initial half velocities
    for n in range(0, DOFs, DIMENSIONS):
        start, end = n, n+DIMENSIONS
        vhalf[start:end, 0] = v[start:end, 0] + ( dt / 2 ) * tot_force(r[:, 0], n)
    
    # loop for the rest of the timesteps
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
            vhalf[start:end, i] = vhalf[start:end, i-1] + k
            
    return r, v


