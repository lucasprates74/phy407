import numpy as np
"""
Study the quantum system with potential well of the form V(x)= ax/l within the interval
[0, l] and infinte potential everywhere else.

Authors: Lucas Prates
"""

# Define the constants of the problem here
a = 10  # eV
l = 5  # angstroms
e_mass = 9.1094e-31  # kg
e_charge = 1.6022e-19  # Coulombs
pi = np.pi
hbar = 6.582119569e-16  # eV seconds

def same_parity(m, n):
    return 

def Hmn(m, n):
    """
    A function which defines the element in the mth row and nth collumn of the 
    Hamiltonian matrix for this problem.
    """
    if m == n:
        return 0.5 * a + (pi * hbar * m / l) ** 2 / (2 * e_mass)
    elif m % 2 != n % 2:
        return - 8 * a * m * n / (pi * (m ** 2 - n ** 2)) ** 2
    else:
        return 0