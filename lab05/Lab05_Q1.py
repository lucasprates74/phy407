import numpy as np
from scipy.constants import speed_of_light as c
import matplotlib.pyplot as plt
import Lab05_MyFunctions as myf

# define relavant constants
MASS = 1 # kg
SPRING_CON = 12 # N / m
OMEGA = np.sqrt(SPRING_CON / MASS)

xc = c / OMEGA # this is the amplitude at which the classical oscillator reaches the speed of light


def acceleration(x, v):
    """
    Function that gives the acceleration of the relativistic spring for a given
    position and velocity.
    """
    return -(SPRING_CON / MASS) * x (1 - v ** 2 / c ** 2) ** (3/2)

if __name__ == '__main__':

    # part a
    x0, x1, x2 = 1, xc, 10 * xc

    t, x, v = myf.EulerCromer(x0, 0, acceleration, 100, 1000)

    plt.plot(t, x)
    plt.show()