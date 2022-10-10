"""
Q3 code. Analyzes the surface level pressure (SLP) anomalies and its Fourier wavemodes for a fixed latitude of 50 degrees for 120 days (starting Jan. 1, 2015).  
Author: Sam De Abreu
"""
import numpy as np
from matplotlib.pyplot import contourf, xlabel, ylabel, title, colorbar
import matplotlib.pyplot as plt
import Lab05_MyFunctions as myf

def waveform(A, phi, lamb, m):
    """
    The basic waveform of the Fourier decomp. in the longitudinal direction for the SLP. Supplied from the lab handout
    """
    return A*np.cos(lamb*m+phi)

def extract_and_plot(Longitude, Times, SLP, m):
    """
    Compute the mth Fourier component of the longitude as a function of time and plot the component in a latitude-time cross section contour plot
    """
    SLP_fft = np.fft.fft(SLP, axis=1)[:, m] # Compute the Fourier transform in the longitudinal direction. Get the mth coefficient as a function of time
    A = np.abs(SLP_fft) # Get the amplitude [A=sqrt(real**2+imag**2)]
    phi = np.arctan2(np.imag(SLP_fft), np.real(SLP_fft)) # Compute the phase
    waveforms = []
    # Compute the waveform for time and longitude. Need to be in the form waveform[day][longitude]
    for i in range(len(SLP)):
        waveforms.append(waveform(A[i], phi[i], Longitude, m)) 
    # Plot contours
    contourf(Longitude, Times, waveforms)
    xlabel('longitude(degrees)')
    ylabel('days since Jan. 1 2015')
    title('$m={0}$ Fourier component of SLP anomaly data (hPa)'.format(m))
    colorbar()
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


