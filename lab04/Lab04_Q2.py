import numpy as np
from numpy.linalg import eigvalsh, eigh
import matplotlib.pyplot as plt
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

def psi(x, eigenvector):
    """
    Generate eigenfunctions for this problem given an eigenvector
    """
    # normalize eigenvector
    eigenvector = eigenvector / np.sqrt(sum(eigenvector ** 2))

    N = len(eigenvector)
    val = 0
    for n in range(N):
        val += eigenvector[n] * np.sin(n * pi * x / l)
    
    return val



if __name__ == '__main__':
    # part c
    print(eigvalsh(H(10)))

    # part d
    H_arr = H(100)
    print(eigvalsh(H_arr)[:10])

    # part e
    eigenvalues, eigenvectors = eigh(H_arr)  # 
    
    eigenvectors = eigenvectors.transpose()  # 

    x = np.linspace(0, l, 100)  # 

    ground_state, first_excited, second_excited = eigenvectors[0:3]

    wfn0 = psi(x, ground_state)
    wfn1 = psi(x, first_excited)
    wfn2 = psi(x, second_excited)

    plt.plot(x, np.abs(wfn0) ** 2)
    plt.plot(x, np.abs(wfn1) ** 2)
    plt.plot(x, np.abs(wfn2) ** 2)
    plt.show()