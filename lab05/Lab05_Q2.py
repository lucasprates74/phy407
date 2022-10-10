""" The scipy.io.wavfile allows you to read and write .wav files """
from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt
import Lab05_MyFunctions as myf
plt.rcParams.update({'font.size': 16}) # change plot font size

# read the data into two stereo channels
# sample is the sampling rate, data is the data in each channel,
1
# dimensions [2, nsamples]
sample, data = read('GraviteaTime.wav')
# sample is the sampling frequency, 44100 Hz
# separate into channels
channel_0 = data[:, 0]
channel_1 = data[:, 1]
N_Points = len(channel_0)
# ... do work on the data...

time = np.linspace(0,len(data)//sample, len(data))  # create time array for data

# part b plot two channels
figj, axj = plt.subplots(2, 1, figsize=(8,8))
plt.suptitle('GraviteaTime Wave Form')
axj[0].plot(time, channel_0)
axj[0].set_ylabel('Channel 0')
axj[1].plot(time, channel_1)
axj[1].set_ylabel('Channel 1')
plt.xlabel('Time (sec)')

# optional part c
# axj[0].set_xlim(0, 0.02)
# axj[1].set_xlim(0, 0.02)

plt.tight_layout()
plt.show()

# part d

def filter_func(freq, amplitudes, cutoff):

    cutoff_index = np.argwhere(freq < cutoff).ravel()[-1]
    
    filtered = amplitudes.copy()
    filtered[cutoff_index:] = 0
    return filtered


def fourier_filter(channel, name):
    coefficients = np.fft.rfft(channel)
    ang_freq = 2 * np.pi * np.arange(len(coefficients)) / 8  

    cutoff = 880 # Hz
    filtered = filter_func(ang_freq, coefficients, 2 * np.pi *cutoff)
    
    figj, axj = plt.subplots(2, 1, figsize=(8,8))
    plt.suptitle('GraviteaTime FFT for Channel {}'.format(name))
    axj[0].plot(ang_freq, np.abs(filtered))
    axj[0].set_ylabel('Amplitude')
    axj[1].plot(ang_freq, np.abs(filtered))
    axj[1].set_ylabel('Filtered Amplitude')
    plt.xlabel('Angular Frequency (s$^{-1}$)')
    plt.tight_layout()
    plt.show()

    filtered_signal = np.fft.irfft(filtered)

    figj, axj = plt.subplots(2, 1, figsize=(8,8))
    plt.suptitle('GraviteaTime Wave Form for Channel {}'.format(name))
    axj[0].plot(time, channel)
    axj[0].set_ylabel('Original Signal')
    axj[1].plot(time, filtered_signal)
    axj[1].set_ylabel('Filtered Signal')
    axj[0].set_xlim(0, 0.05)
    axj[1].set_xlim(0, 0.05)    
    plt.xlabel('Time (sec)')
    plt.tight_layout()
    plt.show()
    
    return filtered_signal

channel_0_out = fourier_filter(channel_0, 0)
channel_1_out = fourier_filter(channel_1, 1)
# this creates an empty array data_out with the same shape as "data"
# (2 x N_Points) and the same type as "data" (int16)
data_out = np.empty(data.shape, dtype = data.dtype)
# fill data_out
data_out[:, 0] = channel_0_out
data_out[:, 1] = channel_1_out
write('Filtered_GraviteaTime.wav', sample, data_out)