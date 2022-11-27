import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16}) # change plot font size
"""
This code serves to integrate the function x ** (-1/2) / (1 + np.exp(x)) using both the mean value method
and the importance sampling method with probability density 1 / (2 * np.sqrt(x)).
Authors: Lucas Prates
"""

N = 10**5  # num sample points 
a, b = 0, 1

def f(x):
    """The function to be integrated"""
    return x ** (-1/2) / (1 + np.exp(x))


def p(x):
    """The probability denstiy function"""
    return 1 / (2 * np.sqrt(x))


def transform(x):
    """Transforms a uniformly distributed array into one with distribuiton p(x)"""
    return x ** 2


def mean_value_method():
    """
    Finds the value of the integral of f(x) from a to b using the mean value method.
    """
    x = np.random.uniform(low=a, high=b, size=(N,))

    return (b - a) * np.sum(f(x)) / N


def importance_sampling_method():
    """
    Finds the value of the integral of f(x) from a to b using the importance sampling method,
    with the normalized weight function p(x)
    """
    x = transform(np.random.uniform(low=a, high=b, size=(N,)))

    # p(x) is already normalized, no need to multiply by anything
    return np.sum(f(x) / p(x)) / N

# create arrays to store each attempt of each method
attempts = 100
mvms = np.zeros(attempts)
isms = np.zeros(attempts)

for n in range(attempts):
    mvms[n] = mean_value_method()
    isms[n] = importance_sampling_method()

bins = np.linspace(0.8, 0.88, 11)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Mean Value Results")
plt.hist(mvms, bins=bins, rwidth=0.8)
plt.xlabel("$I$")
plt.ylabel("Occurances")

plt.subplot(1,2,2)
plt.title("Importance Sampling Results")
plt.hist(isms, bins=bins, rwidth=0.8)
plt.xlabel("$I$")
plt.savefig("q3", dpi=300, bbox_inches='tight')
plt.show()