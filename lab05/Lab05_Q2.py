"""
Q2 code. Filters frequencies above a specified cutoff from an audio file using Fourier Transforms, making the most fire mixtapes.
Authors: Sam De Abreu & Lucas Prates
"""
from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt
import Lab05_MyFunctions as myf
plt.rcParams.update({'font.size': 19}) # change plot font size

def read_file(name):
    """
    Read .wav file and store channel 0 & 1 data into separate variables. Return both channels, original data (for shape) and sample rate (Hz/sample)
    """
    sample, data = read(name)
    channel_0 = data[:, 0]
    channel_1 = data[:, 1]
    return channel_0, channel_1, data, sample

def filter_func(freq, amplitudes, cutoff):
    """Filter function for cutting off frequencies above the specified cutoff of a fourier transform"""

    cutoff_index = np.argwhere(freq < cutoff).ravel()[-1] # Find index where frequency passes cutoff
    
    filtered = amplitudes.copy() # Get a deep copy
    filtered[cutoff_index:] = 0 # Set frequencies above cutoff to 0
    return filtered

def fourier_filter(channel, name, cutoff=880):
    """
    Takes in channel and filters the audio such that it removes all audio above the cut off frequency
    """
    coefficients = np.fft.rfft(channel) # Perform fourier transform on channel
    ang_freq = 2 * np.pi * np.arange(len(coefficients)) / 8  # Get the angular frequency for as per the index of the fourier coefficients (Period=8s)

    filtered = filter_func(ang_freq, coefficients, 2 * np.pi *cutoff) # Filter the fourier transform
    
    # Plot original Fourier transform of waveform and filtered Fourier transform of waveform 
    figj, axj = plt.subplots(2, 1, figsize=(8,8))
    plt.suptitle('GraviteaTime FFT for Channel {}'.format(name))
    axj[0].plot(ang_freq, np.abs(coefficients))
    axj[0].set_ylabel('Amplitude')
    axj[0].set_xticks([0, 7e4, 1.4e5])
    axj[1].plot(ang_freq, np.abs(filtered))
    axj[1].set_xticks([0, 7e4, 1.4e5])
    axj[1].set_ylabel('Filtered Amplitude')
    plt.xlabel('Angular Frequency (s$^{-1}$)')
    plt.tight_layout()
    plt.savefig('Q2c_Fourier_channel_{0}.png'.format(name), dpi=300, bbox_inches='tight')

    filtered_signal = np.fft.irfft(filtered) # Inverse Fourier transform on filtered Fourier transform, to get the final filitered waveform

    # Plot original waveform and filtered waveform
    figj, axj = plt.subplots(2, 1, figsize=(8,8))
    plt.suptitle('GraviteaTime Wave Form for Channel {}'.format(name))
    axj[0].plot(time, channel)
    axj[0].set_ylabel('Original Signal')
    axj[1].plot(time, filtered_signal)
    axj[1].set_ylabel('Filtered Signal')
    axj[0].set_xlim(0, 0.05)
    axj[1].set_xlim(0, 0.05)    
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('Q2c_Waveform_channel_{0}.png'.format(name), dpi=300, bbox_inches='tight')
    
    return filtered_signal

def write_file(data, channel_0_out, channel_1_out, sample):
    """
    Write to .wav file
    """
    data_out = np.empty(data.shape, dtype = data.dtype)
    data_out[:, 0] = channel_0_out
    data_out[:, 1] = channel_1_out
    write('Filtered_GraviteaTime.wav', sample, data_out)


if __name__ == '__main__':
    # Getting/reading the data and storing it
    channel_0, channel_1, data, sample = read_file('GraviteaTime.wav')
    
    N = len(data)
    time = np.linspace(0, N//sample, N)  # create time array for data
    
    # part b , plotting original wave forms
    figj, axj = plt.subplots(2, 1, figsize=(8,8))
    plt.suptitle('GraviteaTime WaveForm for both Channels')
    axj[0].plot(time, channel_0)
    axj[0].set_ylabel('Channel 0 Waveform')
    axj[1].plot(time, channel_1)
    axj[1].set_ylabel('Channel 1 Waveform')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('Q2b.png', dpi=300, bbox_inches='tight')

    # part d
    channel_0_out = fourier_filter(channel_0, 0)
    channel_1_out = fourier_filter(channel_1, 1)

    # part e
    write_file(data, channel_0_out, channel_1_out, sample)