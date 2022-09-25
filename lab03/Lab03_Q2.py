import numpy as np
import scipy.constants as CON
import matplotlib.pyplot as plt
import Lab03_myFunctions as myf

MASS = 1 # kg
SPRING_CON = 12 # N / m
OMEGA = np.sqrt(SPRING_CON / MASS)
c = CON.c

CLASSICAL_LIM = 2 * np.pi / OMEGA

def g(x, x0):
    numerator = SPRING_CON * (x0 ** 2 - x ** 2) * (2 * MASS * c ** 2 + SPRING_CON * (x0 ** 2 - x ** 2) / 2)
    denominator = 2 * (MASS * c ** 2 + SPRING_CON * (x0 ** 2 - x ** 2) / 2) ** 2

    return c * np.sqrt(numerator / denominator)

def T(x0, N):
    x, w = myf.gaussxwab(N, 0, x0)
    
    g_arr = g(x, x0)
    integral = sum(4 * w / g_arr)
    return integral, x, w, g_arr

def T_parte(x0, N):
    x, w = myf.gaussxw(N)
    integral_arr = np.zeros(len(x0))

    for i in range(len(x0)):
        print(i)
        xi = x0[i]
        xnew, wnew = 0.5 * xi * x + 0.5 * xi, 0.5* xi * w
        g_arr = g(xnew, xi)
        
        integral_arr[i] = sum(4 * wnew / g_arr)
    return integral_arr

if __name__ == '__main__':
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
    plt.show()

    plt.plot(x8, 4*w8/g_arr8, linestyle='none', marker='.', label='Weighted Integrand (N=8)')
    plt.plot(x16, 4*w16/g_arr16, linestyle='none', marker='.', label='Weighted Integrand (N=16)')
    plt.legend()
    plt.show()

    # part c
    xc = c / OMEGA

    # part d
    T200 = T(x0, 200)[0]
    err200 = np.abs(T200 - CLASSICAL_LIM)
    print('Fractional Error for N=200:', err200 / CLASSICAL_LIM)


    # part e
    x0 = np.arange(1, 10 * xc, 10 ** 5)
    
    plt.plot(x0, T_parte(x0, 200), linestyle='none', marker='.')
    plt.show()