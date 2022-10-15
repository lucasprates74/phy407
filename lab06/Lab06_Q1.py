"""
Q1 code. Solves and plots the trajectory of two particles interacting via the Lennard-Jones potential.
Author: Sam De Abreu
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16}) # change plot font size

N = 100 # Number of steps

def f(r): 
    """
    Returns the force felt by both particles as a 4d array
    """
    # Import the components of both particles' locations
    x1 = r[0]
    y1 = r[1]
    x2 = r[2]
    y2 = r[3]
    
    # Compute the distance between the particles
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Compute the force
    factor = (24 / dist ** 14) * (2 - dist ** 6)
    fx = (x1-x2) * factor
    fy = (y1-y2) * factor
    return np.array([fx, fy, -fx, -fy])

def solve(r1, r2, v1, v2, dt=0.01):
    """
    Solve the force equation using the Verlet method with the initial conditions: r1=(x1,y1), r2=(x2,y2), v1=(xdot1,ydot1), and v2=(xdot2,ydot2)
    """
    # Set initial conditions
    r = np.zeros((4, N))
    r[0][0], r[1][0] = r1
    r[2][0], r[3][0] = r2
    v = np.zeros((4, N))
    v[0][0], v[1][0] = v1
    v[2][0], v[3][0] = v2
    v_half = np.zeros((4, N))
    # Compute the initial v(t+h/2)
    v_half[:, 0] = v[:, 0]+dt/2*f(r[:, 0])
    # Integrate using the Verlet method
    for i in range(N-1): 
        r[:, i+1] = r[:, i]+dt*v_half[:, i]
        k = dt*f(r[:, i+1])
        v[:, i+1] = v_half[:, i]+0.5*k
        v_half[:, i+1] = v_half[:, i]+k # Compute the v(t+3h/2) for the next iteration
    return r, v

if __name__ == '__main__':
    # part b
    initial_conditions = [[(4, 4), (5.2, 4), (0, 0), (0, 0)], [(4.5, 4), (5.2, 4), (0, 0), (0, 0)], [(2, 3), (3.5, 4.4), (0, 0), (0, 0)]]
    for m in range(3):
        r1_sol, v1_sol = solve(*initial_conditions[m]) # Solve the problem for the initial conditions
        for i in range(N):
            # Plot the trajectory solution
            plt.plot(r1_sol[0][i], r1_sol[1][i], marker='.', linestyle='None', color='blue', alpha=i/N)
            plt.plot(r1_sol[2][i], r1_sol[3][i], marker='.', linestyle='None', color='red', alpha=i/N)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.grid()
        plt.title('Trajectory for $N=2$ particles with {0} Initial Conditions'.format({1: '1st', 2: '2nd', 3: '3rd'}[m+1]))
        plt.savefig('Q1b_{0}.png'.format(m+1), dpi=300, bbox_inches='tight')
        plt.clf()
        
        # Analyzes the time series behaviour 
        #plt.plot(np.arange(0, N, 1), r1_sol[0], color='blue')
        #plt.plot(np.arange(0, N, 1), r1_sol[2], color='red')
        #plt.show()