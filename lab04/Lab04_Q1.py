"""
Q1 code. Compares three algorithms: Gauss elimination, Partial pivoting, and LU in terms of their error and time to run.
Author: Sam De Abreu
"""
# Import necessary stuff
import numpy as np
import SolveLinear as sl
import matplotlib.pyplot as plt
from time import time
plt.rcParams.update({'font.size': 16}) # change plot font size

# Constants
N = 300 # Maximum input size

def generate_data():
    """
    Generates and returns time to run and error for each of the three algorithms for matrices and column vectors of size n=2 to n=N. 
    """
    times = {'Gauss': [], 'Partial': [], 'LU': []}
    method_acc = {'Gauss': [], 'Partial': [], 'LU': []}
    funcs = {'Gauss': sl.GaussElim, 'Partial': sl.PartialPivot, 'LU': np.linalg.solve}
    for n in range(2, N):
        # Generate random vector v and matrix A
        v = np.random.rand(n)
        A = np.random.rand(n, n)
        # Evaluate different methods in time complexity and accuracy
        for s in ['Gauss', 'Partial', 'LU']: # The three desired methods
            start = time()
            x = funcs[s](A, v) # Solve Matrix equation using desired method
            end = time()
            times[s].append(end-start) # Record time
            method_acc[s].append(np.mean(abs(v-np.dot(A, x)))) # Record error
    return times, method_acc

if __name__ == '__main__':
    # Plot the error
    N_values = np.arange(2, N) 
    times, method_acc = generate_data()
    for s in ['Gauss', 'Partial', 'LU']:
        plt.semilogy(N_values, method_acc[s], label=s)
    plt.legend()
    plt.xlabel('Matrix and Vector size $N$')
    plt.ylabel('Error $\\varepsilon$')
    plt.title('Error Between Different Algorithms across Input Size $N$')
    plt.savefig('Q1Error.png', dpi=300, bbox_inches='tight')
    plt.clf()

    #Plot the time to run
    for s in ['Gauss', 'Partial', 'LU']:
        plt.semilogy(N_values, method_acc[s], label=s)
    plt.legend()
    plt.xlabel('Matrix and Vector size $N$')
    plt.ylabel('Time to Run $t$')
    plt.title('Time to Run between Different Algorithms across Input Size $N$')
    plt.savefig('Q1Times.png', dpi=300, bbox_inches='tight')