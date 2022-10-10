import numpy as np
from scipy.constants import speed_of_light as c
import matplotlib.pyplot as plt
import Lab05_MyFunctions as myf
plt.rcParams.update({'font.size': 16}) # change plot font size
"""
Here we continue our study of the relativistic spring by taking the fourier transforms of 
the spring position datasets for different initial positions.

Author: Lucas Prates
"""
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

def g(x, x0):
    """
    Velocity function from differential equation. x0 is the initial amplitude while x
    is the position array.
    """
    numerator = SPRING_CON * (x0 ** 2 - x ** 2) * (2 * MASS * c ** 2 + SPRING_CON * (x0 ** 2 - x ** 2) / 2)
    denominator = 2 * (MASS * c ** 2 + SPRING_CON * (x0 ** 2 - x ** 2) / 2) ** 2

    return c * np.sqrt(numerator / denominator)

def T(x0, N):
    """
    This function gives the period for a selected value x0. N is the number of 
    sample points in the integral. The integral is estimated using the method
    of Gaussian Quadratures.

    Returns the value of the integral.
    """
    x, w = myf.gaussxwab(N, 0, x0)  # compute sample points and weighs
    
    g_arr = g(x, x0)
    integral = sum(4 * w / g_arr) # compute integral

    return integral


if __name__ == '__main__':

    # part a
    x0_slow, x0_fast, x0_rel = 1, xc, 10 * xc
    
    Tslow = T(x0_slow, 200)  # approximate period for a classical oscillator
    end_slow = 10 * Tslow
    t_slow, x_slow, v_slow = myf.EulerCromer(x0_slow, 0, acceleration, end_slow, 1e-3)

    plt.plot(t_slow, x_slow)
    plt.xlabel('time $t$')
    plt.ylabel('position $x(t)$')
    plt.title('Position vs time for initial amplitude 1 meter')
    plt.show()

    Tfast = T(x0_fast, 200)   # approximate period for an oscillator nearing the speed of light
    end_fast = 10 * Tfast
    t_fast, x_fast, v_fast = myf.EulerCromer(x0_fast, 0, acceleration, end_fast, 1e-5)

    plt.plot(t_fast, x_fast)
    plt.xlabel('time $t$')
    plt.ylabel('position $x(t)$')
    plt.title('Position vs time for initial amplitude $x_c$')
    plt.show()

    Trel = T(x0_rel, 200)   # approximate period for an oscillator nearing the speed of light
    end_rel = 10 * Trel
    t_rel, x_rel, v_rel = myf.EulerCromer(x0_rel, 0, acceleration, end_rel, 1e-4)

    plt.plot(t_rel, x_rel)
    plt.xlabel('time $t$')
    plt.ylabel('position $x(t)$')
    plt.title('Position vs time for initial amplitude $10x_c$')
    plt.show()

    #  part b
    ang_freq_slow, amplitudes_slow = myf.normalized_fft(x_slow, end_slow)
    ang_freq_fast, amplitudes_fast = myf.normalized_fft(x_fast, end_fast)
    ang_freq_rel, amplitudes_rel = myf.normalized_fft(x_rel, end_rel)

    plt.plot(ang_freq_slow, amplitudes_slow, label='1 meter')
    plt.plot(ang_freq_fast, amplitudes_fast, label='$x_c$')
    plt.plot(ang_freq_rel, amplitudes_rel, label='$10x_c$')
    plt.vlines(x=2*np.pi/Tslow, ymin=0, ymax=1, linestyle='--', color='blue')
    plt.vlines(x=2*np.pi/Tfast, ymin=0, ymax=1, linestyle='--', color='orange')
    plt.vlines(x=2*np.pi/Trel, ymin=0, ymax=1, linestyle='--', color='green')
    plt.xlim(0, 6)
    plt.xlabel('Angular frequency $\\omega$')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('FFT of position data')
    plt.show()


    #  part d
    ang_freq_slow, amplitudes_slow = myf.normalized_fft(v_slow, end_slow)
    ang_freq_fast, amplitudes_fast = myf.normalized_fft(v_fast, end_fast)
    ang_freq_rel, amplitudes_rel = myf.normalized_fft(v_rel, end_rel)

    plt.plot(ang_freq_slow, amplitudes_slow, label='1 meter')
    plt.plot(ang_freq_fast, amplitudes_fast, label='$x_c$')
    plt.plot(ang_freq_rel, amplitudes_rel, label='$10x_c$')
    plt.xlim(0, 6)
    plt.xlabel('Angular frequency $\\omega$')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('FFT of velocity data')
    plt.show()