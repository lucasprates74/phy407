"""
Q1 code. Solves Newman's Exercise 8.8: The trajectory of space garbage. We compare RK4 and an adaptive RK4 method on numerically estimating the solution. The code is adapted from Nicolas Grisouard's implementation of RK4 to this problem.
Author: Sam De Abreu  
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from time import time

# Constans and text formatting
ftsz = 16 # Font size
font = {'size': ftsz}  
rc('font', **font) # Change font size
a = 0.0 # The initial time for integration
b = 10.0 # The final time for integration

def rhs(r):
    """ The right-hand-side of the equations
    INPUT:
    r = [x, vx, y, vy] are floats (not arrays)
    note: no explicit dependence on time
    OUTPUT:
    1x2 numpy array, rhs[0] is for x, rhs[1] is for vx, etc"""
    M = 10. # Mass (kg)
    L = 2. # Length of rod (m)
    # Extract position and velocity
    x = r[0]
    vx = r[1]
    y = r[2]
    vy = r[3]
    # Compute the force
    r2 = x**2 + y**2
    Fx, Fy = - M * np.array([x, y], float) / (r2 * np.sqrt(r2 + .25*L**2))
    return np.array([vx, Fx, vy, Fy], float)

def RK4(h):
    """
    Fourth order Runge-Kutta implementation to solve the ball bearing's trajectory. (Author: Nicolas Grisouard). this method also records the time it takes to perform the integration.
    """
    tpoints = np.arange(a, b, h)
    xpoints = []
    vxpoints = []  # the future dx/dt
    ypoints = []
    vypoints = []  # the future dy/dt

    # below: ordering is x, dx/dt, y, dy/dt
    r = np.array([1., 0., 0., 1.], float)
    start = time()
    for _ in tpoints:
        xpoints.append(r[0])
        vxpoints.append(r[1])
        ypoints.append(r[2])
        vypoints.append(r[3])
        r = step(r, h) # RK4 algorithm
    end = time()
    return xpoints, ypoints, end - start  

def step(r, h):
    """
    The basic RK4 algorithm. 
    """
    k1 = h*rhs(r)  # all the k's are vectors
    k2 = h*rhs(r + 0.5*k1)  # note: no explicit dependence on time of the RHSs
    k3 = h*rhs(r + 0.5*k2)
    k4 = h*rhs(r + k3)
    return r + (k1 + 2*k2 + 2*k3 + k4)/6

def RK4_adaptive(h_i, delta):
    """
    Fourth order Runge-Kutta with adaptive step sizes implementation to solve the ball bearing's trajectory. h_i and delta are the inital step size and desired target-error per second, respectively. This method also records the time it takes to perform the integration to solve the system. 
    """
    # Initialize arrays
    xpoints = []
    vxpoints = []  # the future dx/dt
    ypoints = []
    vypoints = []  # the future dy/dt
    time_steps = []
    # Initial conditions
    t = a
    h = h_i
    r = np.array([1., 0., 0., 1.], float)
    xpoints.append(r[0])
    vxpoints.append(r[1])
    ypoints.append(r[2])
    vypoints.append(r[3])
    time_steps.append([t, h])
    # Adaptive RK4 algorithm
    start = time()
    while t < b: # Loop until reached endpoint time b
        r1 = step(step(r, h), h) 
        r2 = step(r, 2*h)
        e_x = 1/30*(r1[0]-r2[0]) # Error in x
        e_y = 1/30*(r1[2]-r2[2]) # Error in y
        rho = 30*h*delta/np.sqrt(e_x**2+e_y**2) # rho as in equation (4) in the handout
        
        if rho >= 1: # If step size is good, go to next 2h. Otherwise, repeat loop with new h
            r = r1
            xpoints.append(r[0])
            vxpoints.append(r[1])
            ypoints.append(r[2])
            vypoints.append(r[3])
            time_steps.append([t, h])
            t += 2*h
        # Compute new step size h
        factor = rho**(1/4)
        if factor > 2:
            factor = 2
        h *= factor
    end = time()
    return xpoints, ypoints, end - start, time_steps

if __name__ == '__main__':
    # Solve the system with RK4 and adaptive RK4
    xpoints, ypoints, runtime = RK4(h=1e-3) 
    xpoints_adapt, ypoints_adapt, runtime_adapt, time_steps = RK4_adaptive(h_i=1e-2, delta=1e-6)

    # part a
    plt.figure()
    plt.plot(xpoints, ypoints, ':', label='RK4')
    plt.plot(xpoints_adapt, ypoints_adapt, ':', label='Adaptive RK4')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend(prop={'size': 11})
    plt.title('Trajectory of a ball bearing around a space rod', fontsize=ftsz)
    plt.axis('equal')
    plt.grid()
    plt.savefig('Garbage.png', dpi=150)
    plt.savefig('Q1a.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # part b
    print('Runtime for RK4: {0}'.format(runtime))
    print('Runtime for adaptive RK4: {0}'.format(runtime_adapt))
    print('Adaptive RK4 is {0} times faster than normal RK4'.format(runtime/runtime_adapt))

    # part c
    time_array = np.array(time_steps)
    start = 5
    fig, axs = plt.subplots(4, 1, figsize=(8,8))
    axs[0].plot(time_array[:, 0][start:], xpoints_adapt[start:])
    axs[0].set_ylabel('$x$ (m)')
    axs[1].plot(time_array[:, 0][start:], ypoints_adapt[start:])
    axs[1].set_ylabel('$y$ (m)')
    axs[2].plot(time_array[:, 0][start:], (np.array(xpoints_adapt[start:])**2+np.array(ypoints_adapt[start:])**2)**(0.5))
    axs[2].set_ylabel('$r$ (m)')
    axs[3].plot(time_array[:, 0][start:], time_array[:, 1][start:])
    axs[3].set_ylabel('Time step $h$ (s)')
    axs[3].set_xlabel('Time $t$ (s)')
    plt.suptitle('Spatial Coordinates $(x,y)$ & Time Step $h$ vs Time $t$')
    plt.tight_layout()
    plt.savefig('Q1c.png', dpi=300, bbox_inches='tight')
