"""
Q2 code. Here we solve the shallow water equations using FTCS. The solutions are plotted and animated.
Author: Sam De Abreu
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
from pylab import clf, plot, xlim, ylim, show, pause, draw
plt.rcParams.update({'font.size': 16}) # change plot font size

# Constants
L = 1 # Domain length [m]
Delta_x = 0.02 # Spatial grid spacing [m]
g = 9.81 # Acceleration due to gravity [ms^{-2}]
H = 0.01 # Domain height [m]
Delta_t = 0.01 # Timestep [s]
J = int(L/Delta_x) # Number of spatial grid points
x_grid = np.linspace(0, L, J) # The spatial grid
T = 4 # Total runtime [s]
N = int(T/Delta_t) + 1 # Number of time steps

# Functions

def u_i(x,t=0):
    """
    The initial condition for the velocity u: u(x, t=0).
    """
    return 0

def eta_i(x,t=0):
    """
    The initial condition for the altitude of the fluid surface eta: eta(x, t=0)
    """
    A = 0.002
    mu = 0.5
    sigma = 0.05
    return H+A*np.exp(-(x-mu)**2/sigma**2)-np.average(A*np.exp(-(x_grid-mu)**2/sigma**2)) # Initial form is a Gaussian wavepacket

def solve_FTCS():
    """
    Solver for the shallow waves equation. We use the FTCS method
    """
    u = np.zeros((N, J)) # u(t,x) := u[t_n][x_j]
    eta = np.zeros((N, J)) # eta(t, x) := eta[t_n][x_j]
    for n in range(-1, N-1): # Loop through all time steps
        for j in range(J): # Loop through all spatial grid points
            if n == -1: # t = 0 (initial condition)
                u[n+1][j] = u_i(x_grid[j])
                eta[n+1][j] = eta_i(x_grid[j])
            else: # t > 0
                if j == 0: # x = 0 (boundary condition)
                    u[n+1][j] = 0
                    eta[n+1][j] = eta[n][j] - Delta_t/Delta_x * (eta[n][j] * (u[n][j+1] - u[n][j]) + u[n][j] * (eta[n][j+1] - eta[n][j]))
                elif j == J-1: # x = L (boundary condition)
                    u[n][j] = 0
                    eta[n+1][j] = eta[n][j] - Delta_t/Delta_x * (eta[n][j] * (u[n][j] - u[n][j-1]) + u[n][j] * (eta[n][j] - eta[n][j-1]))
                else: # Inside domain: x in (0, L)
                    u[n+1][j] = u[n][j] - Delta_t/(2*Delta_x) * (u[n][j] * (u[n][j+1] - u[n][j-1]) + g * (eta[n][j+1] - eta[n][j-1]))
                    eta[n+1][j] = eta[n][j] - Delta_t/(2*Delta_x) * (eta[n][j] * (u[n][j+1] - u[n][j-1]) + u[n][j] * (eta[n][j+1] - eta[n][j-1]))
    return u, eta

def anim(eta):
    """
    To animate the solution produced from solver_FTCS.
    """
    for n in range(0, N):
        clf()
        plot(x_grid, np.ones(len(x_grid))*H, linestyle='dashed', color='black')
        plot(x_grid, np.zeros(len(x_grid)), color='black')
        plot(x_grid, eta[n])
        ylim(0, 0.015)
        xlim(0, L)
        draw()
        pause(0.01)

if __name__ == '__main__':
    # Get solution u and eta
    u, eta = solve_FTCS()
    # Get requested times
    times_ind = [0, int(1/Delta_t), int(4/Delta_t)]
    # Plot the solutions
    for t_ind in times_ind:
        plt.plot(x_grid, np.ones(len(x_grid))*H, linestyle='dashed', color='black', label='$\\eta_b$')
        plt.plot(x_grid, np.zeros(len(x_grid)), color='black', label='$H$')
        plt.plot(x_grid, eta[t_ind], label='$\\eta(x, t={0}$s$)$'.format(t_ind*Delta_t))
        plt.ylim(0, plt.ylim()[1]+H/8)
        plt.xlim(0, L)
        plt.legend()
        plt.ylabel('Altitude $z$ (m)')
        plt.xlabel('$x$ (m)')
        plt.title('$\\eta(x,t)$ Evaluated at $t={0}$s'.format(t_ind*Delta_t))
        plt.grid()
        plt.savefig('Lab08_Q2b_t{0}.png'.format(t_ind*Delta_t), dpi=300, bbox_inches='tight')
        plt.clf()
    
    #anim(eta)
    
    