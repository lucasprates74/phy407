import numpy as np 
"""
Implementation of the integration rules used throughout this project.
Authors: Lucas Prates & Sam De Abreu
"""
def trap_rule(func, a, b, N):
    """
    Integrates func from a to b by approximating with N trapezoids.
    """
    dx = (b - a) / N

    val = 0.5 * (func(a) + func(b))

    for i in range(1, N):
        val += func(a + i*dx)

    return val * dx


def simp_rule(func, a, b, N):
    """
    Integrates func from a to b by approximating with N quadratics. N must be even.
    Note that this is slightly different to the formula in the textbook. The initial value
    is func(a)-func(b) instead of func(a)+func(b), and the even sum runs from 2...N instead of 2...N-2.
    Mathematically, these two changes cancel out. But numerically, there will be a slight deviation
    from the predictions made by the textbook's version of simpson's rule.
    """
    dx = (b - a) / N

    val = func(a) - func(b)

    for i in range(1, N, 2):
        val += 4 * func(a + i * dx) + 2 * func(a + (i + 1) * dx)
        
    return val * dx / 3


def trap_rule_err(func, a, b, N):
    """
    Returns the approximate error in the trapezoid rule approximation. N must be even.
    """
    integral2 = trap_rule(func, a, b, N) 
    integral1 = trap_rule(func, a, b, N // 2)

    return np.abs(integral2 - integral1) / 3

def rel_error(expected, estimated):
    """
    Computes the relative error of the estimated value with the expected value.
    """
    return abs((estimated-expected)/expected)

if __name__ == '__main__':
    print(trap_rule(np.sin, 0, np.pi, 10 ** 6))
    print(simp_rule(np.sin, 0, np.pi, 10 ** 6))