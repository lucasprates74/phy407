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
        a[:, j] = dcst.dst2(f[:, j])

    # Take transform along y
    for i in range(M):
        # DST a[i, :] and set as b[i, :]
        b[i, :] = dcst.dst2(a[i, :])
    return b

def idSSt2(b):
    """ Takes iDST along y, then iDST along x (X = C/S)
    IN: b, the input 2D numpy array
    OUT: f, the 2D inverse-transformed array """
    M = f.shape[0] # Number of rows
    N = f.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    f = np.zeros((M, N)) # Final array
    
    # Take inverse transform along y
    for i in range(M):
        # iDXT b[i,:] and set as a[i,:]
        a[i, :] = dcst.dst2(b[i, :])

    # Take inverse transform along x
    for j in range(N):
        # iDXT a[:,j] and set as f[:,j]
        f[:, j] = dcst.dst2(a[:, j])
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
        a[:, j] = dcst.dst2(f[:, j])

    # Take transform along y
    for i in range(M):
        # DST a[i, :] and set as b[i, :]
        b[i, :] = dcst.dct2(a[i, :])
    return b

def idSSt2(b):
    """ Takes iDST along y, then iDST along x (X = C/S)
    IN: b, the input 2D numpy array
    OUT: f, the 2D inverse-transformed array """
    M = f.shape[0] # Number of rows
    N = f.shape[1] # Number of columns
    a = np.zeros((M, N)) # Intermediate array
    f = np.zeros((M, N)) # Final array
    
    # Take inverse transform along y
    for i in range(M):
        # iDXT b[i,:] and set as a[i,:]
        a[i, :] = dcst.dst2(b[i, :])

    # Take inverse transform along x
    for j in range(N):
        # iDXT a[:,j] and set as f[:,j]
        f[:, j] = dcst.dct2(a[:, j])
    return f


def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w