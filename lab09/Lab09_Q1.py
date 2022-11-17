"""
Q1 code. Here we setup and solve the time-dependent Schrodinger equation for a one-dimensional square well using the Crank-Nicolson method. The resulting solution wavefunction is plotted along with some of its properties.
Author: Sam De Abreu
"""
#Imports
import Lab09_MyFunctions as myf
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as CON
plt.rcParams.update({'font.size': 16}) # change plot font size

# Constants
L = 1e-8 # Well width [m]
m = 9.109e-31 # Electron mass [kg]
P = 1024 # Number of cells in the spatial grid
N = 3000 # Number of time steps
N_int = 300 # Number of steps to use in integration
tau = 1e-18 # Time step [s]
a = L/P # Spatial cell spacing [m]
T = tau*N # Total runtime [s]

# Functions
def normalize(f):
    """
    Computes eqn (3) on a function f from lab handout using Gaussian quadratures. 
    """
    x, w = myf.gaussxwab(N_int, -L/2, L/2)
    s = 0
    for i in range(len(x)):
        s += w[i]*f(x[i])*np.conjugate(f(x[i]))
    return s

def psi_func_i(x):
    """
    The initial condition of the wavefunction psi without normalization
    """
    sigma = L/25
    kappa = 500/L
    x0 = L/5
    return np.exp(-(x-x0)**2/(4*sigma**2)+1j*kappa*x)

psi0 = np.sqrt(1/normalize(psi_func_i)) # Compute the normalization constant (done outside psi_i so normalize() isn't called everytime psi_i() is)
def psi_i(x):
    """
    The initial condition of the wavefunction psi with normalization
    """
    return psi0*psi_func_i(x)

def V(x):
    """
    The potential. We use a one-dimensional square well potential with it being zero in (-L/2, L/2) and inf elsewhere.
    """
    if type(x) is np.ndarray: # Check if input is an array
        temp = []
        for x_i in x: # Loop through each value since we cannot take the truth value of an entire array
            if -L/2 < x_i < L/2:
                temp.append(0)
            else:
                temp.append(1e64)
        return np.array(temp)
    else:
        if -L/2 < x < L/2:
            return 0
        else:
            return 1e64 # A really big number to represent inf (never used, just for show)

def build_H():
    """
    Constructs the discretized Hamiltonian of the system.
    """
    A = -CON.hbar**2/(2*m*a**2)
    S_0 = np.eye(P-1, k=-1)*A
    p = np.arange(1, P, 1)
    S_1 = np.diag(V(p*a-L/2)-2*A, k=0)
    S_2 = np.eye(P-1, k=1)*A
    return S_0 + S_1 + S_2

def solve_system():
    """
    Solve the time-dependent Schrodinger equation using the Crank-Nicolson method for a one-dimensional square well.
    """
    # Setup
    H_d = build_H()  
    x = np.linspace(-L/2, L/2, P-1)
    L_mat = np.eye(P-1, k=0) + 1j*tau/(2*CON.hbar)*H_d
    R_mat = np.eye(P-1, k=0) - 1j*tau/(2*CON.hbar)*H_d
    L_mat_inv = np.linalg.inv(L_mat)
    # Initial condition
    psi = np.zeros((N, P-1), dtype='complex_')
    psi[0] = psi_i(x)
    # Algorithm
    for n in range(N-1):
        # We compute psi^(n+1) = L^(-1) R psi^(n) instead of solving L psi^(n+1) = R psi^(n) since L is constant in time (can be inverted outside of loop), so this is more efficient
        psi[n+1] = np.matmul(L_mat_inv, np.matmul(R_mat, psi[n])) # psi^(n+1) = L^(-1) R psi^(n)
    return psi

def simpson_integrate(f):
    """
    The Simpson integration method. Used for the solution wavefunction since the data spacing is regular, making Gaussian quadratures not an option.
    """
    odd_sum = 0
    even_sum = 0
    for k in range(1, len(f)-1):
        if k % 2 == 0:
            even_sum += 4*f[k]
        else:
            odd_sum += 2*f[k]
    return a/3 * (f[0] + f[-1] + even_sum + odd_sum)

def compute_pos_expec(psi):
    """
    Computes the expectation value of position across time for the solution wavefunction. This computation uses the Simpson method for its integration.
    """
    expec = []
    for n in range(len(psi)):
        expec.append(simpson_integrate(np.conj(psi[n])*x*psi[n]))
    return np.real(expec) # To ensure solution is entirely real (imag part is ~0 anyways)

def compute_energy(psi):
    """
    Computes the energy across time of the solution wavefunction.
    """
    H_d = build_H()
    energy = []
    for n in range(len(psi)):
        f = np.matmul(np.matmul(np.conj(psi[n]), H_d), psi[n]) 
        energy.append(a*f)
    return np.real(energy) # To ensure solution is entirely real (imag part is ~0 anyways)

def compute_norm(psi):
    """
    Computes the normalization across time of the solution wavefunction. This computation uses the Simpson method for its integration. 
    """
    norms = []
    for n in range(len(psi)):
        norms.append(simpson_integrate(np.conj(psi[n]) * psi[n]))
    return np.real(norms) # To ensure solution is entirely real (imag part is ~0 anyways)

if __name__ == '__main__':
    psi = solve_system() # Get the solution wavefunction
    x = np.linspace(-L/2, L/2, P-1) # Spatial coordinates
    time = np.linspace(0, tau*N, N) # Temporal coordinates

    # part b
    # Time slices
    T_ind = [0, int(N/4), int(N/2), int(3*N/4), int(N)-1]
    T_labels = ['0', 'T/4', 'T/2', '3T/4', 'T']
    for i in range(len(T_ind)):
        plt.plot(x/L, np.real(psi[T_ind[i]]), label='$\\psi(x,{0})$'.format(T_labels[i]))
        plt.xlabel('$x/L$')
        plt.ylabel('Re($\\psi$)')
    plt.grid()
    plt.ylim(-5.5e4, 5.5e4)
    plt.xlim(-1/2, 1/2)
    plt.legend(prop={'size': 10}, loc='upper right')
    plt.title('Time slices of $\\psi(x,t)$')
    plt.savefig('Q1bTimeslices.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # Position expectation value
    x_expec = compute_pos_expec(psi)
    plt.plot(time/T, np.array(x_expec)/L)
    plt.xlabel('$t/T$')
    plt.ylabel('$<X>(t)/L$')
    plt.grid()
    plt.title('Expectation Value of Position as a Function of Time')
    plt.savefig('Q1bPosExpec.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # Part c
    # Energy
    energy = compute_energy(psi)
    plt.plot(time/T, energy)
    plt.ylim(0, 1e-16)
    plt.xlabel('$t/T$')
    plt.ylabel('$E(t)$ (J)')
    plt.grid()
    plt.title('Energy of $\\psi(x,t)$ as a Function of Time')
    plt.savefig('Q1cEnergy.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # Normalization
    norms = compute_norm(psi)
    plt.plot(time/T, norms)
    plt.ylim(0.8, 1.2)
    plt.xlabel('$t/T$')
    plt.ylabel('Normalization')
    plt.grid()
    plt.title('Normalization as a Function of Time')
    plt.savefig('Q1cNorm.png', dpi=300, bbox_inches='tight')
    plt.clf()
