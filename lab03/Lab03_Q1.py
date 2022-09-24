import numpy as np 
import matplotlib.pyplot as plt
import Lab03_myFunctions as myf
"""
The purpose of this script is to test the accuracy of the forward difference and central difference
numerical differentiation techniques.

Author: Lucas Prates
"""

a = 0.5  # the value where we will evaluate the derivative of e**-x**2

# generate a stepsize array ranging from 10**-16 to 1, incrementing by multiples of 10
h_arr = np.power(10 * np.ones(17), np.arange(-16, 1)) 

def f(x):
    return np.e ** (- x ** 2)

EXPECTED = -f(a)  # the expected value of the derivative of f at x=0.5



def main(rule):
    """
    Given a numerical differentiation rule, "rule", returns the numerical error
    in the rule for the function f defined above for 17 different step sizes.
    """
    derivative_arr = []

    for h in h_arr:
        derivative_arr.append(rule(f, a, h))
    
    return np.abs(np.array(derivative_arr)-EXPECTED)

    

if __name__ == '__main__':
    forward_err = main(myf.forward_diff)
    central_err = main(myf.central_diff)

    # part a
    print('step size    ', 'error')
    for i in range(len(h_arr)):
        print(h_arr[i], forward_err[i])

    # part b
    plt.plot(h_arr, forward_err, linestyle='none', marker='.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('step size $h$')
    plt.ylabel('Error $\\varepsilon(h)$')
    plt.title('Error in numerical derivative vs step size')
    plt.show()

    # part c
    plt.plot(h_arr, forward_err, linestyle='none', marker='.', label='Forward Difference')
    plt.plot(h_arr, central_err, linestyle='none', marker='.', label='Central Difference')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('step size $h$')
    plt.ylabel('Error $\\varepsilon(h)$')
    plt.title('Error in numerical derivative vs step size')
    plt.show()

    


