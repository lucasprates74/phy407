import numpy as np
from scipy.constants import speed_of_light as c

SPRING_CONST = 1
MASS = 1

def acceleration(x, v):
    return -(SPRING_CONST / MASS) * x (1 - v ** 2 / c ** 2) ** (3/2)

def EulerCromer(x0, v0, func, end_time, num_points):
    dt = end_time / num_points
    t = np.arange(0, end_time, dt)
    x, v = np.zeros(num_points), np.zeros(num_points)

    # set initial conditions
    x[0], v[0] = x0, v0

    for i in range(1, num_points):
        v[i] = func(x[i - 1, v - 1]) * dt + v[i - 1]
        x[i] = v[i] * dt + x[i - 1]

    return t, x, v