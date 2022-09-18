import numpy as np
import Lab02_myFunctions as myf
from time import time

INTEGRAL = np.pi
a, b = 0, 1

def f(x):
    return 4 / (x ** 2 + 1)


# part b
print('Trapezoid rule:', myf.trap_rule(f, a, b, 4))

print('Simpson\'s rule:', myf.simp_rule(f, a, b, 4))

# part c
ACC = 10 ** - 9
def get_rule_performance(rule, acc):
    """
    Returns the number of slices required to evaluate an integral using the 
    method rule to an accuracy of acc. Also returns the time required to evaluate rule
    at this accuracy
    """
    N = 2
    while np.abs(INTEGRAL - rule(f, a, b, N)) > acc:
        N = 2 * N

    start = time()
    rule(f, a, b, N)
    end = time()
    return N, end - start

N_trap, t_trap = get_rule_performance(myf.trap_rule, ACC)

N_simp, t_simp = get_rule_performance(myf.simp_rule, ACC)

print('trapezoid rule: slices = {0}, time = {1}s'.format(N_trap, t_trap))
print('Simpson\'s rule: slices = {0}, time = {1}s'.format(N_simp, t_simp))