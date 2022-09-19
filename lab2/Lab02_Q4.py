import numpy as np
import matplotlib.pyplot as plt


def p(x):
    return (1 - x) ** 8


def q(x):
    """Algebraically expanded form of (1 - x) ** 8. Absolutely horrendous looking"""
    return 1 - 8 * x + 28 * x ** 2 - 56 * x ** 3 + 70 * x ** 4 - 56 * x ** 5 + 28 * x ** 6 - 8 * x ** 7 + x ** 8

if __name__ == '__main__':

    # part a
    start, stop = 0.98, 1
    step = (stop - start) / 500
    x=np.arange(start, stop, step)
    print(x)
    plt.plot(x, p(x), label='p(x)', linestyle='none', marker='.')
    plt.plot(x, q(x), label='q(x)', linestyle='none', marker='.')
    plt.legend()
    plt.show()