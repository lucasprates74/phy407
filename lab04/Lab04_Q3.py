import numpy as np
import matplotlib.pyplot as plt

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

def f(x, c):
    return 1-np.exp(-c*x)


if __name__== '__main__':
    #Exercise 6.10, part a
    print('Solution to equation (within 10^(-6)) with initial guess x_i = {0} and c = {1}: x = {2}'.format(0.5, 2, relaxation(f, 0.5, 1e-6, c=2)))
    # part b
    solutions = []
    c_values = np.arange(0, 3.01, 0.01)
    for c in c_values:
        solutions.append(relaxation(f, 0.5, 1e-6, c)[0])
    plt.plot(c_values, solutions)
    plt.xlabel('Parameter $c$')
    plt.ylabel('Solution $x$')
    plt.title('Bifurcation Plot of Equation (something) With Initial Guess $x_i=0.5$')
    plt.show()

    # Exercise 6.11
    omega = np.arange(0, 1.5, 0.05)
    num_iter = np.zeros(len(omega))
    for i in range(len(omega)):
        num_iter[i] = relaxation(f, 0.5, 1e-6, c=2, omega=omega[i])[1]
    plt.plot(omega, num_iter, linestyle='none', marker='.')
    plt.xlabel('$\\omega$')
    plt.ylabel('Number of iterations')
    plt.title('Number of iterations vs $\\omega$')
    plt.show()