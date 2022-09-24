import numpy as np 
import matplotlib.pyplot as plt
import Lab03_myFunctions as myf
plt.rcParams.update({'font.size': 16}) # change plot font size
"""
Question 3 code. Calculates the nth wavefunction and its <X^2>, <P^2> and E_n as well as plots some relationships.
Author: Sam De Abreu
"""
# Constants 
N = 100 # Number of points for Gaussian quadrature on

def H(n, x):
    """
    Hermite function H_n(x) following the recursive formulation: H_{n}(x)=2xH_{n-1}(x)-2x(n-1)H_{n-2}(x) supplied from the lab handout.
    """
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return 2*x*H(n-1, x)-2*(n-1)*H(n-2, x) # Lab handout equation (9)

def psi(n, x):
    """
    Wavefunction psi_n(x) for the quantum harmonic oscillator where "n" is the energy level.
    """
    coeff = H(n, x)/np.sqrt(np.sqrt(np.pi)*float(2**n)*float(np.math.factorial(n))) # Lab handout equation (8)
    return coeff*np.exp(-x**2/2)

def psi_x(n, x):
    """
    Derivative of the wavefunction psi_n(x) with respect to x.
    """
    if n == 0: # Special case of formula supplied in lab handout since H_{-1}(x) is undefined 
        return -x*np.exp(-x**2/2)/np.pi**0.25
    coeff = (-x*H(n, x)+2*n*H(n-1, x))/np.sqrt(np.sqrt(np.pi)*float(2**n)*float(np.math.factorial(n))) # Lab handout equation (11)
    return coeff*np.exp(-x**2/2)

def expec_x_squared_integrand(n, x):
    """
    Integrand for the <X^2> computation at a specific energy level "n". Note we use the change of variable u = tan(x) to rewrite it from a improper to proper integral.
    """
    return np.tan(x)**2/np.cos(x)**2*abs(psi(n, np.tan(x)))**2

def expec_p_squared_integrand(n, x):
    """
    Integrand for the <P^2> computation at a specific energy level "n". Note we use the change of variable u = tan(x) to rewrite it from a improper to proper integral.
    """
    return abs(psi_x(n, np.tan(x)))**2/np.cos(x)**2

def expec_x_squared(n):
    """
    Computation of <X^2> at energy level "n". We use Gaussian Quadrature (N=100) to compute the integral in the formula (equation (12) from lab handout).
    """
    x, w = myf.gaussxwab(N, -np.pi/2, np.pi/2)
    s = 0
    for i in range(len(x)):
        s += w[i]*expec_x_squared_integrand(n, x[i])
    return s

def expec_p_squared(n):
    """
    Computation of <P^2> at energy level "n". We use Gaussian Quadrature (N=100) to compute the integral in the formula (equation (12) from lab handout).
    """
    x, w = myf.gaussxwab(N, -np.pi/2, np.pi/2)
    s = 0
    for i in range(len(x)):
        s += w[i]*expec_p_squared_integrand(n, x[i])
    return s

def total_energy(n):
    """
    Computation of total energy E_n at energy level "n". 
    """
    return 0.5*(expec_x_squared(n)+expec_p_squared(n)) # Equation (14) from lab handout

def x_uncert(n):
    """
    Computation of uncertainty in X (\Delta X) at energy level "n".
    """
    return np.sqrt(expec_x_squared(n))

def p_uncert(n):
    """
    Computation of uncertainty in P (\Delta P) at energy level "n".
    """
    return np.sqrt(expec_p_squared(n))

if __name__ == '__main__':
    # part a)
    x_values = np.arange(-4, 4, 0.1)
    for i in range(4):
        plt.plot(x_values, psi(i, x_values), label='$\\psi_{{{0}}}(x)$'.format(i))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('$\\psi_n(x)$')
    plt.title('First Four Wavefunctions $\\psi_n(x)$')
    plt.grid()
    plt.savefig('Q3wavefunctions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # part b)
    x_values = np.arange(-10, 10, 0.1)
    plt.plot(x_values, psi(30, x_values), label='$\\psi_{{{0}}}(x)$'.format(30))
    plt.xlabel('x')
    plt.ylabel('$\\psi_{{30}}(x)$')
    plt.title('Quantum Harmonic Oscillator Wavefunction $\\psi_{{30}}(x)$')
    plt.grid()
    plt.savefig('Q3Bigwavefunction.png', dpi=300, bbox_inches='tight')
    plt.show() 

    # part c)
    # Position and momentum plot
    x_values = np.arange(0, 15, 1)
    for i in x_values:
        plt.plot(i, x_uncert(i), color='blue', marker='x', ms=10)
        plt.plot(i, p_uncert(i), color='red', marker='o')
        plt.plot(i, x_uncert(i)*p_uncert(i), color='green', marker='s')
    plt.plot([], [], color='blue', marker='x', label='$\\Delta X$')
    plt.plot([], [], color='red', marker='o', label='$\\Delta P$')
    plt.plot([], [], color='green', marker='s', label='$\\Delta X\\Delta P$')
    plt.plot(x_values, x_values+0.5, color='orange', label='$\\left(n+\\frac{{1}}{{2}}\\right)$')
    plt.xlabel('Energy level $n$')
    plt.ylabel('Uncertainty')
    plt.title('$\\Delta X$, $\\Delta P$ and $\\Delta X \\Delta P$ Across Different $n$')
    plt.legend()
    plt.grid()
    plt.savefig('Q3PosMomUncert.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Energy plot
    for i in x_values:
        plt.plot(i, total_energy(i), marker='o', color='blue')
    plt.plot([], [], color='blue', marker='o', label='$E_{{n}}$')
    plt.plot(x_values, x_values+0.5, color='orange', label='$\\left(n+\\frac{{1}}{{2}}\\right)$')
    plt.xlabel('Energy level $n$')
    plt.ylabel('Total energy $E_n$')
    plt.title('Total Energy $E_n$ Across Different $n$')
    plt.legend()
    plt.grid()
    plt.savefig('Q3Energy.png', dpi=300, bbox_inches='tight')
    plt.show()
    

