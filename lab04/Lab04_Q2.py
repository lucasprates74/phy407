import numpy as np
from numpy.linalg import eigvalsh, eigh
import matplotlib.pyplot as plt
import Lab04_myFunctions as myf
plt.rcParams.update({'font.size': 16}) # change plot font size
"""
Study the quantum system with potential well of the form V(x)= ax/l within the interval
[0, l] and infinte potential everywhere else.

Authors: Lucas Prates
"""

# Define the constants of the problem here
e_charge = 1.6022e-19  # Coulombs
c = 299792458 # meters per second
pi = np.pi
hbar =1.054571817e-34 # eV seconds
e_mass = 9.1094e-31  # kg

# define well parameters
a = 10 * e_charge # Joules
l = 5e-10  # meters

x,w = myf.gaussxwab(200, 0, l) # get integration parameters for later


def Hmn(m, n):
    """
    A function which defines the element in the mth row and nth collumn of the 
    Hamiltonian matrix for this problem. NOte that this matrix is Hermitian as 
    swapping m and n changes nothing.
    """
    if m == n:
        return (0.5 * a + (pi * hbar * m / l) ** 2 / (2 * e_mass)) / e_charge
    elif m % 2 != n % 2:
        return (- 8 * a * m * n / (pi * (m ** 2 - n ** 2)) ** 2) / e_charge
    else:
        return 0

def H(N):
    """
    Returns the square submatrix of the hamiltonian of sidelength N.
    """
    H_arr = np.zeros([N, N])

    for m in range(1, N + 1):
        for n in range(1, N + 1):
            H_arr[m-1, n-1] = Hmn(m, n)
    
    return H_arr

def psi(eigenvector):
    """
    Generate eigenfunctions for this problem given an eigenvector
    """
    N = len(eigenvector)
    val = 0
    for n in range(N):
        val += eigenvector[n] * np.sin(n * pi * x / l)
    
    normalization = np.sqrt(sum(np.abs(val) ** 2 * w))
    return val / normalization

def prob_density(eigenvector):
    return np.abs(psi(eigenvector)) ** 2


if __name__ == '__main__':
    # part c
    eigvals10 = eigvalsh(H(10))

    # part d
    H_arr = H(100)
    eigvals100 = eigvalsh(H_arr)[:10]
    
    print('Energy Level & N=10 Eigenvalues (eV) & N=100 Eigenvalues (eV)\\\\\\hline')
    for i in range(10):
        print('{0} & {1} & {2}\\\\'.format(i, round(eigvals10[i], 3), round(eigvals100[i], 3)))

    # part e
    eigenvalues, eigenvectors = eigh(H_arr)
    
    eigenvectors = eigenvectors.transpose()  # transpose eigenvector matrix so the rows are eigenvectors

    # get the eigenvectors for the three lowest energy states
    ground_state, first_excited, second_excited = eigenvectors[0:3]

    # generate probability densities for the three lowest energy states
    pd0 = prob_density(ground_state)
    pd1 = prob_density(first_excited)
    pd2 = prob_density(second_excited)

    plt.plot(x, pd0, label='Ground')
    plt.plot(x, pd1, label='1st Excited')
    plt.plot(x, pd2, label='2nd Excited')
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('Probability Density $|\\psi_n(x)|^2$')
    plt.title('Probaility Density vs x')
    plt.grid()
    plt.gcf().set_size_inches([10,6])
    plt.savefig('Q4.png', dpi=300, bbox_inches='tight')
    plt.show()