"""
Q1 code. Solves the traveling salesman problem and minimizes some example functions using annealing optimization.
Author: Sam De Abreu
"""
# Imports
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size': 16}) # change plot font size

# Constants
N = 25 # Number of points (for traveling salesman) for Q1a

# Q1a code
def curve_fit_func(x, a, b):
    """
    Function used in curve fitting (tau, D) data
    """
    return a*x+b

def mag(x):
    """
    Computes the magnitude of the vector x
    """
    return np.sqrt(x[0]**2 + x[1]**2)

def distance(r):
    """
    Computes the total distance of the traveling salesman problem
    """
    s = 0
    for i in range(N):
        s += mag(r[i+1] - r[i])
    return s

def anneal_Q1a(tau, Tmin, Tmax, loop_seed, pos_seed):
    """
    Solves the traveling salesman problem using Annealing Optimization. This is based off of the textbook's solution (example 10.4). We use an exponential cooling schedule and fix the random number generating for choosing the positions and available paths with pos_seed and loop_seed, respectively, for analysis purposes
    """
    random.seed(pos_seed) # Fix the RNG for picking the positions
    r = np.empty([N+1, 2], float)
    for i in range(N): # Initialize the "random" positions
        r[i,0] = random.random()
        r[i,1] = random.random()
    r[N] = r[0] # Ensure that it ends at the start
    D = distance(r) 
    D_i = D # Initial distance
    t = 0 # Time (for exponential cooling)
    T = Tmax
    random.seed(loop_seed) # Fix the RNG for picking the avialable paths
    while T > Tmin: 
        t += 1
        T = Tmax*np.exp(-t/tau)
        i,j = random.randrange(1, N), random.randrange(1, N) # Choose two random positions
        while i == j: # Ensure they are unique
            i,j = random.randrange(1, N), random.randrange(1, N)
        oldD = D # Store old distance
        # Swap positions
        r[i,0], r[j,0] = r[j,0], r[i,0] 
        r[i,1], r[j,1] = r[j,1], r[i,1]
        D = distance(r) # Compute new distance
        if random.random() >= np.exp(-(D - oldD)/T):
            # Reject the swap
            r[i,0], r[j,0] = r[j,0], r[i,0]
            r[i,1], r[j,1] = r[j,1], r[i,1]
            D = oldD
    return D_i, D, r

def var_D(n):
    """
    Code for analyzing the change in the distance D for different loop_seed. We also plot the varying paths taken.
    """
    D_list = []
    for k in range(n):
        D_i, D, r = anneal_Q1a(tau=1e4, Tmin=1e-3, Tmax=10, loop_seed=k, pos_seed=3) # Compute new D and path for different loop_seed
        D_list.append(D)
        # Plot the corresponding path
        for m in range(len(r)-1): 
            plt.plot(r[m:m+2,0], r[m:m+2,1], marker='o', color='black')
        plt.plot(r[0,0], r[0,1], marker='o', color='red') # Plot the start/end
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Path Solution {2} ($D_{{inital}}={0}$, $D_{{final}}={1}$)'.format(round(D_i, 2), round(D, 2), k+1))
        plt.savefig('Q1aSol{0}'.format(k), dpi=300, bbox_inches='tight')
        plt.clf()
    print(D_list)
    print('Average D value is {0} with stdev of {1}'.format(round(np.average(D_list), 3), round(np.std(D_list), 3)))

def var_tau():
    """
    Code for analyzing the change in the distance D for different loop_seed and tau. We plot (tau, D) to analyze any potential relationships.
    """
    tau_list = []
    D_list = []
    taus = [3e3, 5e3, 7e3, 9e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4] # 1e4 + 0.3e3*(k-n//2)
    for k in range(len(taus)):
        tau = taus[k]
        tau_list.append(tau)
        D_i, D, r = anneal_Q1a(tau=tau, Tmin=1e-3, Tmax=10, loop_seed=k, pos_seed=3) # Compute new D and path for different loop_seed and tau
        D_list.append(D)
        print('D value is {0} with tau of {1}'.format(round(D, 3), round(tau, 3)))
    popt, pcov = curve_fit(curve_fit_func, tau_list, D_list, (0, 0)) # Curve fit the (tau, D) to see if the general trend is increasing or decreasing
    # Plot the relationship
    plt.plot(tau_list, D_list, marker='o', label='Data')
    plt.plot(tau_list, curve_fit_func(np.array(tau_list), popt[0], popt[1]), linestyle='dashed', color='k', label='Linear Fit')
    plt.xlabel('Cooling rate $\\tau$')
    plt.ylabel('Solution Distance $D$')
    plt.title('The Solution Distance $D$ Variation with Cooling Rate $\\tau$')
    plt.grid()
    plt.legend()
    plt.savefig('Q1a2.png', dpi=300, bbox_inches='tight')
    plt.clf()


# Q1b and Q1c code

def f1(x, y):
    """
    Function to minimze in Q1b
    """
    return x**2 - np.cos(4*np.pi*x) + (y-1)**2

def f2(x, y):
    """
    Function to minimize in Q1c
    """
    return np.cos(x) + np.cos(np.sqrt(2)*x) + np.cos(np.sqrt(3)*x) + (y-1)**2

def gen_pos(x, y, lb):
    """
    Generates new position (x_d,y_d) using nonuniform Gaussian sampling based off of inputted pair (x,y). We also ensure the newly generated points lie within defined bounds: lb < x_d < 50, -20 < y_d < 20. Used in the Minimize() function
    """
    # See section 10.1.6 for an explanation on the following code
    r = np.sqrt(-2*np.log(1-random.random()))
    theta = random.random() * 2*np.pi
    delta_x, delta_y = r*np.cos(theta), r*np.sin(theta)
    x_d, y_d = x+delta_x, y+delta_y
    # Ensure newly generated position is in bounds
    if lb < x_d < 50 and -20 < y_d < 20: # lb is used as the lower bound differs between Q1b and Q1c. No bounds are defined by the question for Q1b, so we choose large bounds to ensure they don't interfere with the algorithm
        return x_d, y_d
    else:
        return gen_pos(x, y, lb)

def Minimize(f, Tmax, Tmin, tau, question):
    """
    Finds the global minimum of f using annealing optimization with an exponential cooling schedule. This code is used for Q1b and Q1c
    """
    x, y = 2,2 # Initial position
    T = Tmax
    t = 0
    x_list, y_list = [], []
    while T > Tmin:
        t += 1
        T = Tmax * np.exp(-t/tau)
        x_d, y_d = gen_pos(x, y, question['lb']) # Generate new position
        if random.random() <= np.exp(-(f(x_d, y_d) - f(x, y))/T):
            # Go through with new position
            x, y = x_d, y_d 
        x_list.append(x)
        y_list.append(y)
    print('Minimum is estimated to be (x,y) = ({0}, {1})'.format(round(x,4), round(y,4)))
    # Plot
    plt.plot(x_list, y_list, marker='.', linestyle='None')
    plt.plot(x, y, marker='o', color='k')
    plt.plot(question['correct_pos'][0], question['correct_pos'][1], marker='o', color='red')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(question['title'])
    plt.savefig('Q1b_{0}.png'.format(question['title']), dpi=300, bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    # Part a
    var_D(n=10)
    var_tau()

    # Part b
    Minimize(f1, Tmax=10, Tmin=1e-3, tau=1e4, question={'title': 'Trajectory during Annealing Optimization for $f_1(x,y)$', 'correct_pos': (0, 1), 'lb': -100})
    
    # Part c
    Minimize(f2, Tmax=10, Tmin=1e-3, tau=8e4, question={'title': 'Trajectory during Annealing Optimization for $f_2(x,y)$', 'correct_pos': (16, 1), 'lb': 0})