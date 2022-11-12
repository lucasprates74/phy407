import numpy as np
import matplotlib.pyplot as plt
import Lab09_MyFunctions as myf

# define numerical parameters
P = 32  # grid size
tau = 0.01  # time step size
T = 20  # final time
N = int(T // tau) # time steps

time = np.linspace(0, T, N)

# define physical parameters
Lx = 1  # cavity length x
Ly = 1 # cavity length y

J0 = 1  # current amplitude
m = 1
n = 1
omega = 3.75  # driving frequency
c = 1  # speed of light


# create p, q arrays: 
parr, qarr = np.arange(0, P), np.arange(0, P)

J = np.zeros((N, P, P))
for i in range(N):
    for p in parr:
        for q in qarr: 
            t = time[i]
            J[i, p, q] = J0 * np.sin(m * np.pi * p / P) * np.sin(n * np.pi * p / P) * np.sin(omega * t)
            
Jhat = np.zeros((N, P, P))
for i in range(N):
    Jhat[i] = myf.dSSt2(J[i])

def Crank_Nicolson(Ehat0, Xhat0, Yhat0):
    
    # define constants
    Dx = np.pi * c * tau / (2 * Lx)
    Dy = np.pi * c * tau / (2 * Ly)

    # create arrays for fields over time
    E = np.zeros((N, P, P))
    Hx = np.zeros((N, P, P))
    Hy = np.zeros((N, P, P))

    # set initial conditions
    prev_Ehat, prev_Xhat, prev_Yhat = Ehat0, Xhat0, Yhat0

    for i in range(1, N):
        Ehat = ((1 - parr ** 2 * Dx - qarr ** 2 * Dy) * prev_Ehat \
        + 2 * qarr * Dy * prev_Xhat + 2 * qarr * Dx * prev_Yhat + tau * Jhat[i - 1]) / (1 + parr ** 2 * Dx + qarr ** 2 * Dy)

        Xhat = prev_Xhat - qarr * Dy * (Ehat + prev_Ehat)

        Yhat = prev_Yhat - parr * Dx * (Ehat + prev_Ehat)


        E[i] = myf.idSSt2(Ehat)
        Hx[i] = myf.idSCt2(Xhat)
        Hy[i] = myf.idCSt2(Yhat)

    return E, Hx, Hy


if __name__ == '__main__':
    Ehat, Xhat, Yhat = 0,0,0

    E, Hx, Hy = Crank_Nicolson(0,0,0)

    xmid, ymid = 0.5, 0.5
    p, q = int(P * xmid // Lx),int(P * ymid // Ly)
    E_tracer = E[:, p, q]
    Hx_tracer = Hx[:, p, 0]
    Hy_tracer = Hy[:, 0, q]



    plt.plot(time, E_tracer, label='$E(x=0.5, y=0.5)$')
    plt.plot(time, Hx_tracer, label='$H_x(x=0.5, y=0)$')
    plt.plot(time, Hy_tracer, label='$H_y(x=0, y=0.5)$')
    plt.show()