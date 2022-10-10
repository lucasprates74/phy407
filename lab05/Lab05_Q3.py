import numpy as np
from matplotlib.pyplot import contourf, xlabel, ylabel, title, colorbar
import matplotlib.pyplot as plt
import Lab05_MyFunctions as myf

def waveform(A, phi, lamb, m):
    return A*np.cos(lamb*m+phi)

def extract_and_plot(Longitude, Times, SLP, m):
    SLP_fft = np.fft.fft(SLP, axis=1)[:, m]
    A = np.abs(SLP_fft)
    phi = np.arctan2(np.imag(SLP_fft), np.real(SLP_fft))
    waveforms = []
    for i in range(120):
        waveforms.append(waveform(A[i], phi[i], Longitude, m))
    contourf(Longitude, Times, waveforms)
    xlabel('longitude(degrees)')
    ylabel('days since Jan. 1 2015')
    title('$m=3$ Fourier component of SLP anomaly data (hPa)')
    colorbar()
    plt.show()

if __name__ == '__main__':
    SLP = np.loadtxt('SLP.txt') # [day][longitude]
    Longitude = np.loadtxt('lon.txt')
    Times = np.loadtxt('times.txt')

    contourf(Longitude, Times, SLP)
    xlabel('longitude(degrees)')
    ylabel('days since Jan. 1 2015')
    title('SLP anomaly (hPa)')
    colorbar()
    plt.show()

    extract_and_plot(Longitude, Times, SLP, m=3)
    extract_and_plot(Longitude, Times, SLP, m=5)


