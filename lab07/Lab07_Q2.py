import numpy as np
import scipy.constants as CON 
import matplotlib.pyplot as plt

a = 5e-11  # m, Bohr Radius
ev_to_joule = CON.e

def potential(r, joules=False):
    """Return the potential energy of an electron a distance r from a 
    proton in eV."""

    V = - CON.e ** 2 / (4 * np.pi * CON.epsilon_0 * r)
    
    return V / ev_to_joule


def velocity(x, r, ell, E):
    """Returns the velocity field of x=(R, S) given the current values of 
    r, R, S. ell is a parameter which specifies the angular momentum
    of the system, while E specifies the total energy of the system."""
    R, S = x
    Rdot = S 
    Sdot = ell * (ell + 1) * R / r ** 2 - 2 * S / r + 2 * CON.e * ev_to_joule * (potential(r)-E) / CON.hbar
    return np.array([Rdot, Sdot])

def solve(ell, E, rinfty, dr):
    """
    Solve for the radial wavefunction for a specific angular momentum and 
    energy. Return the value of R at rinfty.
    """
    # start integration at one step size since R diverges for r=0
    rstart = dr 
    R0 = 0
    S0 = 1
    
    N = (rinfty - rstart) // dr 
    x= np.zeros((N, 2))
    x[0, :] = R0, S0
    r = np.arange(rstart, rinfty, N)
    for i in np.arange(1, N):
        k1 = dr * velocity(x[i - 1, :], r[i], ell,E)
        k2 = dr * velocity(x[i - 1, :] + 0.5 * k1, r[i] + 0.5 * dr)
        k3 = dr * velocity(x[i - 1, :] + 0.5 * k2, r[i] + 0.5 * dr)
        k4 = dr * velocity(x[i - 1, :] + k3, r[i] + dr)
        x[i, :] = x[i - 1, :] + (k1 + 2 * k2 + 2 * k3 + k4) / 6 
    
    return x

def get_E(ell, rinfty, dr, acc):
    """Use the secant method to find the energy E of the wavefunction R 
    which gives R(dr)=R(rinfty)=0 to an accuracy of acc""" 
    E1 = - 14  # eV
    E2 = 0  # eV 

    # get last value of R in solution
    R2 = solve(ell, E2, rinfty, dr)[-1, 0]

    R2 = solve(E1)
    while abs(E1-E2)>acc:
        R1, R2 = R2, solve(E2)[-1, 0]
        E1, E2 = E2, E2 - R2 * (E2-E1) / (R2-R1)

    return E2