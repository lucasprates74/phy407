import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16}) # change plot font size
"""
Solves the boundary value problem from Q1 using the Gauss Seidel method. 
Author: Lucas Prates
"""
N = 100  # number of points
L = 10  # cm 
precision = 1e-6 #Volts

# get indeces for the boundary conditions
cm_to_index = N // L 
start = cm_to_index * 2  # index of 2 cm
stop = cm_to_index * 8  # index of 8 cm


def Gauss_Seidel(omega=0):
    """
    Solve the boundary value problem in Q1 using the Gauss-Siedel 
    method with over-relaxation. Iterate 
    until each point reaches the given precision.
    """
    # generate potential grid
    V = np.zeros([N, N])
    # prepare boundary conditions
    V[start:stop, start] = 1    # Volts
    V[start:stop, stop] = -1    # Volts 

    err = 1  # set large initial error 
    err_arr = np.zeros([N, N])
    while precision < err:
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if not ((j == start or j == stop) and start <= i < stop):
                    # get the sum of the adjacent points 
                    adjacent_sum = V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1]

                    # replace the array entry with its new value
                    old = V[i, j]
                    V[i, j] = adjacent_sum * (1 + omega) / 4 - omega * old

                    err_arr[i, j] = np.abs(old - V[i, j])
        
        err = np.max(err_arr)

    return V
if __name__ == '__main__':

    
    # V = Gauss_Seidel()    # 1:24 minutes to run
    # V = Gauss_Seidel(omega=0.1)   # 1:12 minutes to run
    V = Gauss_Seidel(omega=0.5)     # 42 seconds to run

    # get position grid
    x, y = np.linspace(0, L, N), np.linspace(0, L, N)
    plt.contourf(x / L, y / L, V)
    plt.xlabel("$x/\\ell$")
    plt.ylabel("$y/\\ell$")
    plt.title("Electric Potential (V)")
    plt.colorbar()
    plt.savefig("Q1", dpi=300, bbox_inches='tight')
    plt.show()