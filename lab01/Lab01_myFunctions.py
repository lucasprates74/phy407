"""
Functions for computing the acceleration of the Earth due to gravitational forces from the Sun.  
Authors: Lucas Prates 
"""
import numpy as np

# constants 
G = 39.5  # AU^3 MSUN^-1 yr^-2
MJUP = 10 ** -3  # MSUN
RJUP = 5.2  # AU
ALPHA = .01  # AU^2  # alpha larger to see relativistic effects

def gravity(x, y):
    """
    Computes the acceleration due to gravity using Newtonian mechanics in AU/yr^2 as a numpy array
    INPUT:
    Position x and y (AU)
    OUTPUT:
    numpy array of the form (x acceleration, y acceleration)
    """
    coords = np.array([x, y])
    r = np.sqrt(x**2 + y**2)
    return -G * coords / r ** 3 

def gravity_rel(x, y):
    """
    Computes the acceleration due to gravity using relativistic mechanics in AU/yr^2 as a numpy array
    INPUT:
    Position x and y (AU)
    OUTPUT:
    numpy array of the form (x acceleration, y acceleration)
    """
    coords = np.array([x, y])
    r = np.sqrt(x**2 + y**2)
    return gravity(x, y) - G * ALPHA * coords / r ** 5