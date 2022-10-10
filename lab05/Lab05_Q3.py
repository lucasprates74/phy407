import numpy as np
from matplotlib.pyplot import contourf, xlabel, ylabel, title, colorbar
import matplotlib.pyplot as plt
import Lab05_MyFunctions as myf

SLP = np.loadtxt('SLP.txt') # [day][longitude]
Longitude = np.loadtxt('lon.txt')
Times = np.loadtxt('times.txt')

contourf(Longitude, Times, SLP)
xlabel('longitude(degrees)')
ylabel('dayts since Jan. 1 2015')
title('SLP anomaly (hPa)')
colorbar()
plt.show()


SLP_fft = np.fft.fft(SLP, axis=1)[:, 4]
SLP_fft_abs = abs(SLP_fft)
a = np.fft.ifft(SLP_fft)
plt.plot(a)

plt.show()

contourf(Longitude, Times, a)
xlabel('longitude(degrees)')
ylabel('dayts since Jan. 1 2015')
title('SLP anomaly (hPa)')
colorbar()
plt.show()


