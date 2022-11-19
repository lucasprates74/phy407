"""
Q2 code. Here we compute the volume of a d-dimensional unit hypersphere using the Monte Carlo integration method. 
Author: Sam De Abreu
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 1000000 # The number of sampled points to use in the Monte-Carlo integration

def compute_volume(d):
    """
    Computes the volume of a d-dimesional unit hypersphere using the Monte Carlo integration method with N uniformly-sampled points.
    """
    s = 0
    for _ in range(N):
        x = np.random.uniform(low=-1, high=1, size=(d,))
        if np.sum(x**2) <= 1:
            s += 1
    return 2**d/N * s

if __name__ == "__main__":
    # Print out result
    print('The approximate volume of the 10-dimensional unit hypersphere is V = {0} units'.format(compute_volume(10)))
