import numpy as np
import matplotlib.pyplot as plt
import Lab02_myFunctions as myf
plt.rcParams.update({'font.size': 16}) # change plot font size
"""
Code for question 4. Here we see the effects of roundoff error by studying the same polynomial
in factored and expanded form.

Authors: Lucas Prates & Sam De Abreu
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
    """
    A raio mathematically equivalent to 1, for u != 0. Used to study round-off error in multiplication and division.
    """
    return  (u ** 8) / ((u ** 4) * (u ** 4))

if __name__ == '__main__':

    # initialize x array
    start, stop = 0.98, 1.02
    step = (stop - start) / 500
    u=np.arange(start, stop, step)

    # evaluate functions
    pvals = p(u)
    qvals = q(u)

    # part a plots
    plt.plot(u, pvals, label='p(u)', linestyle='none', marker='.')
    plt.plot(u, qvals, label='q(u)', linestyle='none', marker='.')
    plt.legend()
    plt.savefig('Q4pq.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # part b
    diff = pvals - qvals # p(u) - q(u) evaluated over the range of u
    plt.plot(u, diff, linestyle='none', marker='.')
    plt.xlabel('$u$')
    plt.ylabel('Numerical Error $\\varepsilon(u)=p(u)-q(u)$')
    plt.title('Numerical Error $\\varepsilon(u)$')
    plt.savefig('Q4pqDiff.png', dpi=300, bbox_inches='tight')
    plt.clf()

    actual_std = np.std(diff, ddof=1) # calculate std in the histogram from before
    actual_mean = np.mean(diff) # Mean of error distribution
    test_value = 1.2 # a value at which to calculate the std of p(u)-q(u)
    expected_std = C * np.sqrt((len(term_arr(test_value))) * (np.mean(term_arr(test_value) ** 2))) # Equation (3) from lan handout
    print('Actual Standard Deviation:', actual_std)
    print('Expected Standard Deviation:', expected_std)

    plt.hist(diff, bins=30, edgecolor='black') # plot histogram
    plt.ylabel('Frequency')
    plt.text(-3e-14,50, '$\\mu=${0}\n$\\sigma=${1}'.format(round(actual_mean, 17), round(actual_std, 17)), fontsize=12.5)
    plt.title('Histogram ($n_{{bins}}=30$) of $\\varepsilon(u)$ over $u\\in[0.98,1.02]$')
    plt.xlabel('Numerical Error $\\varepsilon(u)$')
    plt.savefig('Q4pqHisto.png', dpi=300, bbox_inches='tight')
    plt.clf()



    # part c
    relative_err = myf.rel_error(pvals, qvals) # Compute relative error

    upper = 60 # upper bound for range of u before relative error heavily diverges
    plt.plot(u[:upper], relative_err[:upper], linestyle='none', marker='.')
    plt.xlabel('$u$')
    plt.ylabel('Relative Error $\\varepsilon_{rel}(u)$')
    plt.title('Relative Error vs u')
    plt.savefig('Q4pqRelErr.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # part d
    plt.plot(u, f(u)-1, linestyle='none', marker='.', label='f(x)')
    plt.xlabel('u')
    plt.ylabel('Numerical Error $f(u)-1$')
    plt.title('Numerical Error vs $u$')
    plt.savefig('Q4pqRatio.png', dpi=300, bbox_inches='tight')
    plt.clf()
    