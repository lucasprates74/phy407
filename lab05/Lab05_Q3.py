"""
Q3 code. Analyzes the surface level pressure (SLP) anomalies and its Fourier wavemodes for a fixed latitude of 50 degrees for 120 days (starting Jan. 1, 2015).  
Author: Sam De Abreu
"""
import numpy as np
from matplotlib.pyplot import contourf, xlabel, ylabel, title, colorbar
import matplotlib.pyplot as plt
import Lab05_MyFunctions as myf
plt.rcParams.update({'font.size': 16}) # change plot font size

def extract_and_plot(Longitude, Times, SLP, m):
    """
    Compute the mth Fourier component of the longitude as a function of time and plot the component in a latitude-time cross section contour plot
    """
    SLP_fft = np.fft.rfft(SLP, axis=1) # Take the Fourier transform in the longitudinal direction
    SLP_fft[:, :m] = 0 # Set all coefficients before the mth to 0 
    SLP_fft[:, m+1:] = 0 # Set all coefficients after the mth to 0
    SLP_comp = np.fft.irfft(SLP_fft, axis=1) # Compute the inverse Fourier transform
    # Plot contours
    contourf(Longitude, Times, SLP_comp)
    xlabel('longitude(degrees)')
    ylabel('days since Jan. 1 2015')
    title('$m={0}$ Fourier component of SLP anomaly data (hPa)'.format(m))
    colorbar()
    plt.gcf().set_size_inches(10,6)
    plt.savefig('Q3anomaly_m{0}.png'.format(m), dpi=300, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    # Import data
    SLP = np.loadtxt('SLP.txt') # [day][longitude]
    Longitude = np.loadtxt('lon.txt')
    Times = np.loadtxt('times.txt')

    # Basic SLP contour plot
    contourf(Longitude, Times, SLP)
    xlabel('longitude(degrees)')
    ylabel('days since Jan. 1 2015')
    title('SLP anomaly (hPa)')
    colorbar()
    plt.savefig('Q3anomaly.png', dpi=300, bbox_inches='tight')
    plt.clf()
    #part a
    extract_and_plot(Longitude, Times, SLP, m=3)
    extract_and_plot(Longitude, Times, SLP, m=5)


