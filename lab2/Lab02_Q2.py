import numpy as np
import Lab02_myFunctions as myf
from time import time
"""
Q2 code. Tests the accuracy and efficiency of the trapezoid and simposon's rule for evaluating the integral
from 0 to 1 of 4/(x**2 + 1).
Authors: Lucas Prates
"""

EXPECTED = np.pi  # expected value of the integral
ACC = 10 ** - 9  # this constant represents the accuracy to which we want to compute the integral in b
a, b = 0, 1  # set the upper and lower bounds for the integral

def f(x):
    """This is the function we want to integrate"""
    return 4 / (x ** 2 + 1)

def get_rule_performance(rule):
    """
    Returns the number of slices required to evaluate an integral using the 
    method rule to an accuracy of acc. Also returns the time required to evaluate rule
    at this accuracy
    """

    # double the value of N until the integral is evaluated to the desired accuracy
    N = 2
    while np.abs(EXPECTED - rule(f, a, b, N)) > ACC:
        N = 2 * N

    # this section is used to measure how much time it takes to evaluate the integral at 
    # the desired accuracy. We evaluate it many times because time() is not very precise on 
    # shorter time scales.
    attempts = 1000 
    start = time()
    for i in range(attempts):
        rule(f, a, b, N)
    end = time()
    return N, (end - start) / attempts

if __name__ == '__main__':
    # part b
    print('Trapezoid rule:', myf.trap_rule(f, a, b, 4))
    print('Simpson\'s rule:', myf.simp_rule(f, a, b, 4))

    # part c
    N_trap, t_trap = get_rule_performance(myf.trap_rule)
    N_simp, t_simp = get_rule_performance(myf.simp_rule)

    print('trapezoid rule: slices = {0}, time = {1}s'.format(N_trap, t_trap))
    print('Simpson\'s rule: slices = {0}, time = {1}s'.format(N_simp, t_simp))