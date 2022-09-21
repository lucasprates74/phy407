import numpy as np
import matplotlib.pyplot as plt
"""
Code for question 4. Here we see the effects of roundoff error by studying the same polynomial
in factored and expanded form.

Authors: Lucas Prates
"""
C = 10 ** - 16

def p(x):
    return (1 - x) ** 8


def q(x):
    """Algebraically expanded form of (1 - x) ** 8. Absolutely horrendous looking"""
    return 1 - 8 * x + 28 * x ** 2 - 56 * x ** 3 + 70 * x ** 4 - 56 * x ** 5 + 28 * x ** 6 - 8 * x ** 7 + x ** 8

def f(x):
    return  (x ** 8) / ((x ** 4) * (x ** 4))

if __name__ == '__main__':

    # initialize x array
    start, stop = 0.98, 1.02
    step = (stop - start) / 500
    x=np.arange(start, stop, step)

    # evaluate functions
    pvals = p(x)
    qvals = q(x)

    # part a
    plt.plot(x, pvals, label='p(x)', linestyle='none', marker='.')
    plt.plot(x, qvals, label='q(x)', linestyle='none', marker='.')
    plt.legend()
    plt.show()

    # part b
    diff = pvals - qvals
    plt.plot(x, diff, linestyle='none', marker='.')
    plt.show()

    plt.hist(diff, bins=30, edgecolor='black')
    plt.show()

    actual_std = np.std(diff, ddof=1)
    expected_std = C * np.sqrt(len(diff) * np.mean(diff ** 2))
    print('Expected Standard Deviation:', actual_std)
    print('Actual Standard Deviation:', expected_std)


    # part c
    relative_err = np.abs(pvals - qvals) / np.abs(pvals)

    upper=30
    plt.plot(x[:upper], relative_err[:upper], linestyle='none', marker='.')
    plt.show()

    # part d
    # plt.plot(x, linestyle='none', marker='.', label='constant value 1')
    plt.plot(x, f(x)-1, linestyle='none', marker='.', label='f(x)')
    plt.legend()
    plt.show()