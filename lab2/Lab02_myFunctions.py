import numpy as np 

def integral_trap(func, a, b, N):
    """
    Integrates func from a to b by approximating with N trapezoids.
    """
    dx = (b - a) / N

    val = 0.5 * (func(a) + func(b))

    for i in range(1, N):
        val += func(a + i*dx)

    return val * dx


def integral_simp(func, a, b, N):
    """
    Integrates func from a to b by approximating with N quadratics.
    """
    dx = (b - a) / N

    val = func(a) - func(b)

    for i in range(1, N, 2):
        val += 4 * func(a + i * dx) + 2 * func(a + (i + 1) * dx)
        
    return val * dx / 3

if __name__ == '__main__':
    print(integral_trap(np.sin, 0, np.pi, 10 ** 6))
    print(integral_simp(np.sin, 0, np.pi, 10 ** 6))