import numpy as np 
"""
Implementation of the integration rules used throughout this project.
Authors: Lucas Prates
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
    Integrates func from a to b by approximating with N quadratics.
    """
    dx = (b - a) / N

    val = func(a) - func(b)

    for i in range(1, N, 2):
        val += 4 * func(a + i * dx) + 2 * func(a + (i + 1) * dx)
        
    return val * dx / 3

if __name__ == '__main__':
    print(trap_rule(np.sin, 0, np.pi, 10 ** 6))
    print(simp_rule(np.sin, 0, np.pi, 10 ** 6))