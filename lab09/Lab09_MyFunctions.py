import dcst
import numpy as np
from numpy import ones,copy,cos,tan,pi,linspace

def dSSt2(f):
    """ Takes DST along x, then DST along y (X = C/S)
    IN: f, the input 2D numpy array
    OUT: b, the 2D transformed array """
    M = f.shape[0] # Number of rows
    N = f.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    b = np.zeros((M, N)) # Final array

    # Take transform along x
    for j in range(N):
        # DST f[:, j] and set as a[:, j]
        a[:, j] = dcst.dst(f[:, j])

    # Take transform along y
    for i in range(M):
        # DST a[i, :] and set as b[i, :]
        b[i, :] = dcst.dst(a[i, :])
    return b

def idSSt2(b):
    """ Takes iDST along y, then iDST along x (X = C/S)
    IN: b, the input 2D numpy array
    OUT: f, the 2D inverse-transformed array """
    M = b.shape[0] # Number of rows
    N = b.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    f = np.zeros((M, N)) # Final array
    
    # Take inverse transform along y
    for i in range(M):
        # iDXT b[i,:] and set as a[i,:]
        a[i, :] = dcst.dst(b[i, :])

    # Take inverse transform along x
    for j in range(N):
        # iDXT a[:,j] and set as f[:,j]
        f[:, j] = dcst.dst(a[:, j])
    return f

def dSCt2(f):
    """ Takes DST along x, then DCT along y (X = C/S)
    IN: f, the input 2D numpy array
    OUT: b, the 2D transformed array """
    M = f.shape[0] # Number of rows
    N = f.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    b = np.zeros((M, N)) # Final array

    # Take transform along x
    for j in range(N):
        # DST f[:, j] and set as a[:, j]
        a[:, j] = dcst.dst(f[:, j])

    # Take transform along y
    for i in range(M):
        # DST a[i, :] and set as b[i, :]
        b[i, :] = dcst.dct(a[i, :])
    return b

def idSCt2(b):
    """ Takes iDCT along y, then iDST along x (X = C/S)
    IN: b, the input 2D numpy array
    OUT: f, the 2D inverse-transformed array """
    M = b.shape[0] # Number of rows
    N = b.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    f = np.zeros((M, N)) # Final array
    
    # Take inverse transform along y
    for i in range(M):
        # iDXT b[i,:] and set as a[i,:]
        a[i, :] = dcst.dct(b[i, :])

    # Take inverse transform along x
    for j in range(N):
        # iDXT a[:,j] and set as f[:,j]
        f[:, j] = dcst.dct(a[:, j])
    return f

def dCSt2(f):
    """ Takes DCT along x, then DST along y (X = C/S)
    IN: f, the input 2D numpy array
    OUT: b, the 2D transformed array """
    M = f.shape[0] # Number of rows
    N = f.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    b = np.zeros((M, N)) # Final array

    # Take transform along x
    for j in range(N):
        # DST f[:, j] and set as a[:, j]
        a[:, j] = dcst.dct(f[:, j])

    # Take transform along y
    for i in range(M):
        # DST a[i, :] and set as b[i, :]
        b[i, :] = dcst.dst(a[i, :])
    return b

def idCSt2(b):
    """ Takes iDST along y, then iDCT along x (X = C/S)
    IN: b, the input 2D numpy array
    OUT: f, the 2D inverse-transformed array """
    M = b.shape[0] # Number of rows
    N = b.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    f = np.zeros((M, N)) # Final array
    
    # Take inverse transform along y
    for i in range(M):
        # iDXT b[i,:] and set as a[i,:]
        a[i, :] = dcst.dst(b[i, :])

    # Take inverse transform along x
    for j in range(N):
        # iDXT a[:,j] and set as f[:,j]
        f[:, j] = dcst.dct(a[:, j])
    return f