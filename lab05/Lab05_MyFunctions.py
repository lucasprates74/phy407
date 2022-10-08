import numpy as np

def EulerCromer(x0, v0, func, end_time, num_points):
    """
    Solves the ode system 

    xdot = v
    vdot = func(x, v)

    given initial conditions x0, v0 for the time range 0 <= t <= end_time using
    the Euler Cromer methond with num_points samples.
    """
    dt = end_time / num_points
    t = np.arange(0, end_time, dt)
    x, v = np.zeros(num_points), np.zeros(num_points)

    # set initial conditions
    x[0], v[0] = x0, v0

    for i in range(1, num_points):
        v[i] = func(x[i - 1, v - 1]) * dt + v[i - 1]
        x[i] = v[i] * dt + x[i - 1]

    return t, x, v