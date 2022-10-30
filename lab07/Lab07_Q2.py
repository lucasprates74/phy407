import numpy as np
import scipy.constants as CON 
import matplotlib.pyplot as plt
"""
Solve the boundary value problem for the radial component of the wavefunction of a hydrogen atom.
Authors: Lucas Prates
"""

plt.rcParams.update({'font.size': 16}) # change plot font size

a = 5e-11  # m, Bohr Radius
ev_to_joule = CON.e  # Joules / eV

# set initial conditions
R0 = 0
S0 = 1

def potential(r):
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
    Sdot = ell * (ell + 1) * R / r ** 2 - 2 * S / r + 2 * CON.electron_mass * ev_to_joule * (potential(r)-E) * R / CON.hbar ** 2

    return np.array([Rdot, Sdot])


def energy(n):
    """Returns the expected eigenvalue for the nth energy level."""
    return -13.6 / n ** 2


def solve(ell, E, r, dr):
    """
    Solve for the radial wavefunction for a specific angular momentum and 
    energy. Return the r and R arrays for plotting.
    """
    # set initial conditions for array
    N = len(r)
    x= np.zeros((2, N))
    x[:, 0] = R0, S0

    # RK4 algorithm
    for i in np.arange(1, N):
        k1 = dr * velocity(x[:, i - 1], r[i - 1], ell, E)
        k2 = dr * velocity(x[:, i - 1] + 0.5 * k1, r[i - 1] + 0.5 * dr, ell, E)
        k3 = dr * velocity(x[:, i - 1] + 0.5 * k2, r[i - 1] + 0.5 * dr, ell, E)
        k4 = dr * velocity(x[:, i - 1] + k3, r[i - 1] + dr, ell, E)
        x[:, i] = x[:, i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6 
    
    # get R data from array
    R = x[0]
    return R


def normalize(r, R, dr):
    """Integrate probability density using trapezoid rule, then return the 
    R normalized R array."""
    # get probability density
    density =  np.abs(R) ** 2

    # trapezoid rule
    normalization = (np.sum(density) + 0.5 * (density[-1] - density[0])) * dr
    
    return R / np.sqrt(normalization)


def solve_BVP(ell, n, acc=1e-3, dr = 0.002 * a, rinfty = 20 * a, plot = False):
    """Use the secant method to find the energy E of the wavefunction R 
    which gives R(dr)=R(rinfty)=0 to an accuracy of acc. 
    
    ell: angular momentum quantum number
    n: energy level
    acc: target accuracy of secant method
    dr: RK4 step size
    rinfty: right bound
    """

    # set bounds for RK4 method and stepsize 
    rstart = dr 

    # generate r array
    r = np.arange(rstart, rinfty, dr)

    # set a finite upper bound for plotting and normalization due to divergence issues
    N = len(r)
    finite_bound = N // 2  
    
    # set initial values for secant method
    E1 = - 15 / n ** 2  # eV
    E2 = -13 / n ** 2  # eV 

    # get last value of R in solution
    R2 = solve(ell, E1, r, dr)[-1]
    
    # secant method
    while abs(E1-E2) > acc:
        R = solve(ell, E2, r, dr)
        R1, R2 = R2, R[-1]
        E1, E2 = E2, E2 - R2 * (E2-E1) / (R2-R1)
    

    # plot the wfn
    if plot == True:
        R = normalize(r[:finite_bound], R[:finite_bound], dr)
        plt.plot(r[:finite_bound]/ a, R, label='$n$={0} and $\\ell$={1}'.format(n, ell))

    return E2


if __name__ == '__main__':
    qnums = [(0,1), (0, 2), (1, 2)]

    # set the default values for the parameters
    dr, rinfty, acc = (0.002 * a, 20 * a, 1e-3)
    
    # generate latex tables for b
    energies=[]
    for qnum in qnums:
        ell, n = qnum
        
        expected = round(energy(n), 6)  # get expceted energy

        print('Modified parameter & Energy(eV) & Error(eV) & Fractional Error\\\\\\hline')
        E = round(solve_BVP(ell, n), 6)  # solve the boundary value problem
        err = round(abs(E - expected), 6)
        frac_err = round(err / abs(expected), 6)
        print('None & ', E, ' & ', err, ' & ', frac_err, '\\\\\\hline')

        E = round(solve_BVP(ell, n, dr=dr/2), 6)  # solve the boundary value problem
        err = round(abs(E - expected), 6)
        frac_err = round(err / abs(expected), 6)
        print('$h \\rightarrow h/2$ & ', E, ' & ', err, ' & ', frac_err, '\\\\\\hline')

        E = round(solve_BVP(ell, n, rinfty=2*rinfty), 6)  # solve the boundary value problem
        err = round(abs(E - expected), 6)
        frac_err = round(err / abs(expected), 6)
        print('$r_{\\infty}\\rightarrow 2r_{\\infty}$ & ', E, ' & ', err, ' & ', frac_err, '\\\\\\hline')

        E = round(solve_BVP(ell, n, acc=acc/10), 6)  # solve the boundary value problem
        err = round(abs(E - expected), 6)
        frac_err = round(err / abs(expected), 6)
        print('$\\epsilon \\rightarrow \\epsilon/10$ & ', E, ' & ', err, ' & ', frac_err, '\\\\\\hline')

        E = solve_BVP(ell, n, dr=dr/10, plot=True)
        energies.append(E)

    # make table of energies for the params I actually want
    print('$(n,\\ell)$ & Energy(eV) & Error(eV) & Fractional Error\\\\\\hline')
    for i in range(len(qnums)):
        ell, n = qnums[i]
        E = round(energies[i], 6)
        expected = round(energy(n), 6)
        err = round(abs(E - expected), 6)
        frac_err = round(err / abs(expected), 6)
        print('({0}, {1}) &'.format(n, ell), E, ' & ', err, ' & ', frac_err, '\\\\\\hline')

    # make plot for c
    plt.title('Radial Wavefunction')
    plt.legend()
    plt.xlabel(r'$r / a$')
    plt.ylabel(r'$R(r)$')
    plt.savefig('Q2', dpi=300, bbox_inches='tight')
    plt.show()