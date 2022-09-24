"""
Q3 code. Implements functions used in calculating the total energy per unit area emitted by a black body following Stefan’s law. Then compares numerically estimated constants (integral and Stefan-Boltzman constant) to actual values.
Author: Sam De Abreu
"""
import numpy as np
import Lab02_myFunctions as myf
import scipy.constants as CON

# Define fundamental constants
h = CON.Planck # Planck's constant (J*s)
k = CON.Boltzmann # Boltzman's constant (J/K)
c = CON.speed_of_light # Speed of light (m/s)

# Define integration constants
N = 10**3 # Number of slices
a = 10**(-3) # Start point of integral
b = 50 #End point of integral

#Define function used in computing W(T)'s integral and W(T) itself
def integrand(x):
    """
    Function used in integrand of the integral computed for W(T)
    """
    return x**3/(np.exp(x)-1)

def W(T):
    """
    Computes total energy per unit area emitted by a black body following Stefan’s law as a function of temperature T. The integration method employed is Simpson's method
    """
    integral = myf.simp_rule(integrand, a, b, N)
    C_1 = 2*np.pi*k**4*T**4/(h**3*c**2)
    return C_1*integral

if __name__ == '__main__':
    # 3 b), Error estimate
    expected_value = np.pi**4/15
    estimated_value = myf.simp_rule(integrand, a, b, N)
    relative_error = (estimated_value-expected_value)/expected_value
    print('Integral value\nExpected value: {0}\nEstimated value: {1}\nRelative error: {2}'.format(expected_value, estimated_value, relative_error))
    
    # 3 c), Error estimate
    expected_value = CON.Stefan_Boltzmann
    estimated_value = W(1)
    relative_error = (estimated_value-expected_value)/expected_value
    print('\nStefan-Boltzman constant\nExpected value: {0}\nEstimated value: {1}\nRelative error: {2}'.format(expected_value, estimated_value, relative_error))