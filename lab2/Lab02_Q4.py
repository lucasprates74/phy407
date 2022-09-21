import numpy as np
import matplotlib.pyplot as plt

C = 10 ** - 16

def p(x):
    return (1 - x) ** 8


def q(x):
    """Algebraically expanded form of (1 - x) ** 8. Absolutely horrendous looking"""
    return 1 - 8 * x + 28 * x ** 2 - 56 * x ** 3 + 70 * x ** 4 - 56 * x ** 5 + 28 * x ** 6 - 8 * x ** 7 + x ** 8

if __name__ == '__main__':

    # initialize x array
    start, stop = 0.98, 1
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

    # expected_std = np.std(diff, ddof=1)
    # sigma = C * np.sqrt(len(diff) * np.mean(diff ** 2))
    # print('Standard Deviation:', expected_std)
    # print('Sigma:', sigma)


    # part c
    relative_err = np.abs(pvals - qvals) / np.abs(pvals)

    upper=60
    plt.plot(x[:upper], relative_err[:upper], linestyle='none', marker='.')
    plt.show()