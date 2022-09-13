import numpy as np

# constants 
G = 39.5  # AU^3 MSUN^-1 yr^-2
MJUP = 10 ** -3  # MSUN
RJUP = 5.2  # AU
ALPHA = .01  # AU^2  # alpha larger to see relativistic effects

def gravity(x, y):
    # returns the acceleration due to gravity in AU/yr^2 as a numpy array
    coords = np.array([x, y])
    r = np.sqrt(x**2 + y**2)
    return -G * coords / r ** 3

def gravity_rel(x, y):
    # returns the acceleration due to relativistic gravity in AU/yr^2 as a numpy array
    coords = np.array([x, y])
    r = np.sqrt(x**2 + y**2)
    return gravity(x, y) - G * ALPHA * coords / r ** 5