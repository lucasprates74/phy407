import numpy as np
import scipy.constants as CON
import matplotlib.pyplot as plt
import Lab03_myFunctions as myf
plt.rcParams.update({'font.size': 16}) # change plot font size
"""
This file is used to study the period of a relativistic pendulum.

Author: Lucas Prates
"""

# define relavant constants
MASS = 1 # kg
SPRING_CON = 12 # N / m
OMEGA = np.sqrt(SPRING_CON / MASS)
c = CON.c

CLASSICAL_LIM = 2 * np.pi / OMEGA  # expected value of integral for small x0

def g(x, x0):
    """
    Velocity function from differential equation. x0 is the initial amplitude while x
    is the position array.
    """
    numerator = SPRING_CON * (x0 ** 2 - x ** 2) * (2 * MASS * c ** 2 + SPRING_CON * (x0 ** 2 - x ** 2) / 2)
    denominator = 2 * (MASS * c ** 2 + SPRING_CON * (x0 ** 2 - x ** 2) / 2) ** 2

    return c * np.sqrt(numerator / denominator)

def T(x0, N):
    """
    This function gives the period for a selected value x0. N is the number of 
    sample points in the integral. The integral is estimated using the method
    of Gaussian Quadratures.

    Returns the value of the integral, the sample points, weights, and the integrand 
    values.
    """
    x, w = myf.gaussxwab(N, 0, x0)  # compute sample points and weighs
    
    g_arr = g(x, x0)
    integral = sum(4 * w / g_arr) # compute integral

    return integral, x, w, g_arr

def T_array(x0, N):
    """
    Returns the period as a function of some set of initial amplitudes x0.
    N is the number of sample points.
    """
    # here we compute x, w for the sample interval [-1, 1], and later
    # stretch and shift them using the required formulae as this is less
    # computationally intensive
    x, w = myf.gaussxw(N)
    integral_arr = np.zeros(len(x0))

    # compute integral for each value of i
    for i in range(len(x0)):
        xi = x0[i]
        xnew, wnew = 0.5 * xi * x + 0.5 * xi, 0.5* xi * w
        g_arr = g(xnew, xi)
        
        integral_arr[i] = sum(4 * wnew / g_arr)
    return integral_arr

if __name__ == '__main__':
    print('Expected Value: ', CLASSICAL_LIM)
    # part a
    x0 = 0.01 # m
    T8, x8, w8, g_arr8 = T(x0, 8) 
    T16, x16, w16, g_arr16 = T(x0, 16)
    err8, err16 = np.abs(T8 - CLASSICAL_LIM), np.abs(T16 - CLASSICAL_LIM)

    print('N=8:')
    print('Period Estimate: ', T8, 's, Period Err: ', err8, 's, Fractional Error', err8/T8)
    print('Period Estimate: ', T16, 's, Period Err: ', err16, 's, Fractional Error', err16/T16)

    # part b
    plt.plot(x8, 4/g_arr8, linestyle='none', marker='.', label='Integrand (N=8)')
    plt.plot(x16, 4/g_arr16, linestyle='none', marker='.', label='Integrand (N=16)')
    plt.legend()
    plt.title('Integrand vs x')
    plt.xlabel('x')
    plt.ylabel('Integrand')
    plt.grid()
    plt.gcf()
    plt.savefig('Q2b.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.plot(x8, 4*w8/g_arr8, linestyle='none', marker='.', label='Weighted Integrand (N=8)')
    plt.plot(x16, 4*w16/g_arr16, linestyle='none', marker='.', label='Weighted Integrand (N=16)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Weighted integrand')
    plt.title('Weighted integrand vs x')
    plt.grid()
    plt.gcf()
    plt.savefig('Q2bweight.png', dpi=300, bbox_inches='tight')
    plt.show()

    # part c
    xc = c / OMEGA # this is the amplitude at which the classical oscillator reaches the speed of light

    # part d
    T200 = T(x0, 200)[0]
    err200 = np.abs(T200 - CLASSICAL_LIM)
    print('Fractional Error for N=200:', err200 / CLASSICAL_LIM)


    # part e
    x0 = np.arange(1, 10 * xc, 10 ** 6)
    
    plt.plot(x0, T_array(x0, 200))
    plt.xlabel('Initial Amplitude $x_0$')
    plt.ylabel('Period $T$')
    plt.title('Period vs Initial Amplitude')
    plt.grid()
    plt.gcf()
    plt.savefig('Q2e.png', dpi=300, bbox_inches='tight')
    plt.show()