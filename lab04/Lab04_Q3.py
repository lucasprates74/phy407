import numpy as np
import matplotlib.pyplot as plt

def relaxation(func, x_i, epsilon, c=1):
    error = 1e2
    while error >= epsilon:
        x = func(x_i, c)
        error = abs(x-x_i)
        x_i = x
    return x_i

def func1(x, c):
    return 1-np.exp(-c*x)


if __name__== '__main__':
    #Exercise 6.10, part a
    print('Solution to equation (within 10^(-6)) with initial guess x_i = {0} and c = {1}: x = {2}'.format(0.5, 2, relaxation(func1, 0.5, 1e-6, c=2)))
    # part b
    solutions = []
    c_values = np.arange(0, 3.01, 0.01)
    for c in c_values:
        solutions.append(relaxation(func1, 0.5, 1e-6, c))
    plt.plot(c_values, solutions)
    plt.xlabel('Parameter $c$')
    plt.ylabel('Solution $x$')
    plt.title('Bifurcation Plot of Equation (something) With Initial Guess $x_i=0.5$')
    plt.show()

