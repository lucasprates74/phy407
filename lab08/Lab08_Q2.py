import numpy as np
import matplotlib.pyplot as plt
from pylab import clf, plot, xlim, ylim, show, pause, draw
plt.rcParams.update({'font.size': 16}) # change plot font size

L = 1 # Domain length [m]
Delta_x = 0.02 # Grid spacing [m]
g = 9.81 # Acceleration due to gravity [ms^{-2}]
H = 0.01 # Domain height [m]
Delta_t = 0.01 # Timestep [s]
J = int(L/Delta_x) # Number of grid points
x_grid = np.linspace(0, L, J)
T = 4 # Total runtime [s]
N = int(T/Delta_t) + 1

# Functions

def u_i(x,t=0):
    return 0

def eta_i(x,t=0):
    A = 0.002
    mu = 0.5
    sigma = 0.05
    exp_term = A*np.exp(-(x-mu)**2/sigma**2)
    return H+A*np.exp(-(x-mu)**2/sigma**2)-np.average(A*np.exp(-(x_grid-mu)**2/sigma**2))

def solve_FTCS():
    u = np.zeros((N, J)) # u(t,x) := u[t][x]
    eta = np.zeros((N, J))
    for n in range(-1, N-1):
        for j in range(J):
            if n == -1: # t = 0
                u[n+1][j] = 0
                eta[n+1][j] = eta_i(x_grid[j])
            else:
                if j == 0: # BC: To the left (x=0)
                    u[n+1][j] = 0
                    eta[n+1][j] = eta[n][j] - Delta_t/Delta_x * (eta[n][j] * (u[n][j+1] - u[n][j]) + u[n][j] * (eta[n][j+1] - eta[n][j]))
                elif j == J-1: # BC: To the right (x=L)
                    u[n][j] = 0
                    eta[n+1][j] = eta[n][j] - Delta_t/Delta_x * (eta[n][j] * (u[n][j] - u[n][j-1]) + u[n][j] * (eta[n][j] - eta[n][j-1]))
                else:
                    u[n+1][j] = u[n][j] - Delta_t/(2*Delta_x) * (u[n][j] * (u[n][j+1] - u[n][j-1]) + g * (eta[n][j+1] - eta[n][j-1]))
                    eta[n+1][j] = eta[n][j] - Delta_t/(2*Delta_x) * (eta[n][j] * (u[n][j+1] - u[n][j-1]) + u[n][j] * (eta[n][j+1] - eta[n][j-1]))
    return u, eta

if __name__ == '__main__':
    u, eta = solve_FTCS()
    times = [0, int(1/Delta_t), int(4/Delta_t)]
    for t in times:
        plt.plot(x_grid, np.ones(len(x_grid))*H, linestyle='dashed', color='black')
        plt.plot(x_grid, np.zeros(len(x_grid)), color='black')
        plt.plot(x_grid, eta[t])
        plt.ylim(0, plt.ylim()[1]+H/8)
        plt.xlim(0, L)
        plt.ylabel('$\\eta(x, t=4$s$)$ (m)')
        plt.xlabel('$x$ (m)')
        plt.title('$\\eta(x,t)$ Evaluated at $t={0}$s'.format(t))
        plt.grid()
        plt.savefig('Lab08_Q2b_t{0}.png'.format(t), dpi=300, bbox_inches='tight')
        plt.clf()
    
    for n in range(0, N):
        clf()
        plot(x_grid, np.ones(len(x_grid))*H, linestyle='dashed', color='black')
        plot(x_grid, np.zeros(len(x_grid)), color='black')
        plot(x_grid, eta[n])
        ylim(0, 0.015)
        xlim(0, L)
        #ylabel('$\\eta(x, t=4$s$)$ (m)')
        #plt.xlabel('$x$ (m)')
        #plt.title('$\\eta(x,t)$ Evaluated at $t={0}$s'.format(t))
        #plt.grid()
        draw()
        pause(0.01)