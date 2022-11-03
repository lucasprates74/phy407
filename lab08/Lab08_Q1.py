import numpy as np



def Gauss_Seidel(phi, precision, omega):
    """
    Given a grid phi with some boundary conditions, use the Gauss-Siedel 
    method with over-relaxation to to solve the Laplace equation. Iterate 
    until each point reaches the given precision.
    """
    N = len(phi)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            # get the sum of the adjacent points 
            adjacent_sum = phi[i+1, j] + phi[i-1][j] + phi[i, j+1] + phi[i, j-1]

            # replace the array entry with its new value
            phi[i, j] = adjacent_sum * (1 + omega) / 4 - omega * phi[i, j]

if __name__ == '__main__':
    npoints = 100
    L = 10  # cm 
    
    # get position grid
    position_grid = np.array([np.linspace(0, L, npoints), np.linspace(0, L, npoints)])

    # generate potential grid
    V = np.zeros([npoints, npoints])

    # get indeces for the boundary conditions
    cm_to_index = npoints // L 
    start = cm_to_index * 2  # index of 2 cm
    stop = cm_to_index * 8  # index of 8 cm

    # prepare boundary conditions
    V[start, start:stop] = 1    # Volts
    V[stop, start:stop] = -1    # Volts 
    