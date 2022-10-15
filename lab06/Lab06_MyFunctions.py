from ssl import VERIFY_X509_PARTIAL_CHAIN
import numpy as np

DIMENSIONS = 2

def force(r1, r2): 
    # Return the force on particle 1 at r1 due to particle 2 at r2
    print(r1)
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
    DOFs = len(r0)  # get the number of degrees of freedom
    num_steps = int(tstop // dt)

    r = np.zeros((DOFs, num_steps))
    v = np.zeros((DOFs, num_steps))
    
    r[:, 0]=r0
    v[:, 0]=v0

    for part_num in range(0, DOFs, DIMENSIONS):
        rpart = r[part_num:part_num+2]
        vpart = v[part_num:part_num+2]

        vhalf = vpart[0] + ( dt / 2 ) * tot_force(r[:, 0], part_num)

        for i in range(1, num_steps):
            rpart[:, i] = rpart[:, i-1] + dt * vhalf
            k = dt * tot_force(r[:, i], part_num)
            vpart[:, i] = vhalf + 0.5 * k
            vhalf += k

    return r, v