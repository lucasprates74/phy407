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
    return -(SPRING_CON / MASS) * x * (1 - v ** 2 / c ** 2) ** (3/2)

def normalized_fft(data):
    amplitudes = np.abs(np.fft.rfft(data))
    return amplitudes / max(amplitudes)


if __name__ == '__main__':

    # part a
    x0_slow, x0_fast, x0_rel = 1, xc, 10 * xc
    
    T = 2 * np.pi / OMEGA  # approximate period for a classical oscillator
    t_slow, x_slow, v_slow = myf.EulerCromer(x0_slow, 0, acceleration, 10 * T, 10 ** 3)

    plt.plot(t_slow, x_slow)
    plt.show()

    T = 2 * np.pi / OMEGA  # approximate period for an oscillator nearing the speed of light
    t_fast, x_fast, v_fast = myf.EulerCromer(x0_fast, 0, acceleration, 10 * T, 10 ** 5)

    plt.plot(t_fast, x_fast)
    plt.show()

    T = 4 * 10 * xc / c  # approximate period for an oscillator nearing the speed of light
    t_rel, x_rel, v_rel = myf.EulerCromer(x0_rel, 0, acceleration, 10 * T, 10 ** 6)

    plt.plot(t_rel, x_rel)
    plt.show()

    #  part b
    amplitudes_slow = normalized_fft(x_slow)[:100]
    amplitudes_fast = normalized_fft(x_fast)[:100]
    amplitudes_rel = normalized_fft(x_rel)[:100]
    plt.plot(amplitudes_slow, label='1m')
    plt.plot(amplitudes_fast, label='xc')
    plt.plot(amplitudes_rel, label='10xc')
    plt.legend()
    plt.show()