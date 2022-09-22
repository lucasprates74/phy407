import numpy as np
import matplotlib.pyplot as plt
import Lab02_myFunctions as myf
"""
Code for question 4. Here we see the effects of roundoff error by studying the same polynomial
in factored and expanded form.

Authors: Lucas Prates
"""
C = 10 ** - 16

def p(u):
    return (1 - u) ** 8


def q(u):
    """Algebraically expanded form of (1 - u) ** 8. Absolutely horrendous looking"""
    return 1 - 8 * u + 28 * u ** 2 - 56 * u ** 3 + 70 * u ** 4 - 56 * u ** 5 + 28 * u ** 6 - 8 * u ** 7 + u ** 8

def term_arr(u):
    """
    A numpy array which gives p(u)-q(u) when summed. This function exists only to make error
    propagation easier.
    """
    return np.array([p(u), -1, 8 * u, -28 * u ** 2, 56 * u ** 3, -70 * u ** 4, 56 * u ** 5, -28 * u ** 6, 8 * u ** 7, -u ** 8])

def f(u):
    return  (u ** 8) / ((u ** 4) * (u ** 4))

if __name__ == '__main__':

    # initialize x array
    start, stop = 0.5, 2
    step = (stop - start) / 500
    u=np.arange(start, stop, step)

    # evaluate functions
    pvals = p(u)
    qvals = q(u)

    # part a
    plt.plot(u, pvals, label='p(u)', linestyle='none', marker='.')
    plt.plot(u, qvals, label='q(u)', linestyle='none', marker='.')
    plt.legend()
    plt.show()

    # part b
    diff = pvals - qvals
    plt.plot(u, diff, linestyle='none', marker='.')
    plt.show()

    plt.hist(diff, bins=30, edgecolor='black')
    plt.show()

    actual_std = np.std(diff, ddof=1) # calculate std in the histogram from before

    test_value = 1.9 # a value at which to calculate the std of p(u)-q(u)
    expected_std = C * np.sqrt((len(term_arr(test_value))) * (np.mean(term_arr(test_value) ** 2)))
    print('Actual Standard Deviation:', actual_std)
    print('Expected Standard Deviation:', expected_std)


    # part c
    relative_err = myf.rel_error(pvals, qvals)

    upper=60
    plt.plot(u[:upper], relative_err[:upper], linestyle='none', marker='.')
    plt.show()

    # part d
    plt.plot(u, f(u)-1, linestyle='none', marker='.', label='f(x)')
    plt.legend()
    plt.show()
