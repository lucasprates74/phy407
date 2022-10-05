"""
Q3 code. Relaxation and Over-relexation method for function solutions to x=f(x) and Binary search for f(x)=0 were implemented.
Authors: Sam De Abreu & Lucas Prates 
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as CON
plt.rcParams.update({'font.size': 16}) # change plot font size

def relaxation(func, x_i, epsilon, c=1, omega=0):
    """
    Finds the solution to x = f(x) using the relaxation algorithm within a value of 
    epsilon. For non-zero values of omega, this function uses overrelaxation. x_i is 
    the initial guess.
    """
    error = 1e2
    num_iter = 0
    while error >= epsilon:
        num_iter += 1
        x = (1 + omega) * func(x_i, c) - omega * x_i
        error = abs(x-x_i)
        x_i = x
    return x_i, num_iter

def binary_search(f, x_1, x_2, epsilon):
    """
    Binary search aglorithm for finding roots of a function (implemented recursively). Only valid for end points x_1 and x_2 such that f(x_1) is a different sign from f(x_2). 
    """
    if f(x_1) * f(x_2) > 0: # If function at end points has same sign, terminate
        return  
    midpoint = 0.5*(x_1+x_2) 
    if f(midpoint) * f(x_1) > 0:
        x_1 = midpoint
    else:
        x_2 = midpoint
    if abs(x_1 - x_2) < epsilon: # Whether answer is within desired accuracy epsilon
        return 0.5*(x_1+x_2)
    else: # Continue searching
        return binary_search(f, x_1, x_2, epsilon)

def f(x, c): # Function used in Exercise 6.10 and 6.11
    return 1-np.exp(-c*x)

def g(x): # Function used in Exercise 6.13
    return 5*np.exp(-x)+x-5

if __name__== '__main__':
    #Exercise 6.10, part a
    print('Solution to equation (within 10^(-6)) with initial guess x_i = {0} and c = {1}: x = {2}'.format(0.5, 2, relaxation(f, 0.5, 1e-6, c=2)[0]))
    # part b
    solutions = []
    c_values = np.arange(0, 3.01, 0.01)
    for c in c_values:
        solutions.append(relaxation(f, 0.5, 1e-6, c)[0])
    plt.plot(c_values, solutions)
    plt.xlabel('Parameter $c$')
    plt.ylabel('Solution $x$')
    plt.title('Bifurcation Plot of $x=1-e^{-cx}$ With Initial Guess $x_i=0.5$')
    plt.savefig('Q3a.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # Exercise 6.11 part b
    num_iter = relaxation(f, 0.5, 1e-6,c=2, omega=0)[1]
    print('The number of iterations for solving x=1-e^(-2x) up to 1e-6: {0}'.format(num_iter))
    # Exercise 6.11 part c
    omega = np.arange(0, 1.5, 0.05)
    num_iter = np.zeros(len(omega))
    for i in range(len(omega)):
        num_iter[i] = relaxation(f, 0.5, 1e-6, c=2, omega=omega[i])[1]
    plt.plot(omega, num_iter, linestyle='none', marker='.')
    plt.xlabel('$\\omega$')
    plt.ylabel('Number of iterations')
    plt.title('Number of iterations vs $\\omega$')
    plt.savefig('Q3b.png', dpi=300, bbox_inches='tight')

    # Exercise 6.13, part b
    sol = binary_search(g, 0.5, 6, 1e-6)
    print("Solution to 5e^(-x) + x - 5 = 0 is x = {0}".format(sol))
    # part c
    sun_lambda = 502e-9 # meters (m)
    b = CON.Planck*CON.speed_of_light/(CON.Boltzmann*sol) # Wien displacement constant (Km)
    print("Estimation of Sun's temperature: T = {0}K".format(b/sun_lambda))