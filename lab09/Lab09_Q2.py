import numpy as np
import matplotlib.pyplot as plt
import Lab09_MyFunctions as myf
plt.rcParams.update({'font.size': 16}) # change plot font size
"""
This code finds the electromagnetic field inside the conducting cavity

Authors: Lucas Prates
"""


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


# create current array
J = np.zeros((N, P, P))
for i in range(N):
    for p in parr:
        for q in qarr: 
            t = time[i]
            J[i, p, q] = J0 * np.sin(m * np.pi * p / P) * np.sin(n * np.pi * q / P) * np.sin(omega * t)


# fourier tranform current at all times
Jhat = np.zeros((N, P, P))
for i in range(N):
    Jhat[i] = myf.dSSt2(J[i])

def Crank_Nicolson():
    
    # define constants
    Dx = np.pi * c * tau / (2 * Lx)
    Dy = np.pi * c * tau / (2 * Ly)

    # create arrays for fields over time. Leave all initial conditions as 0
    E = np.zeros((N, P, P))
    Hx = np.zeros((N, P, P))
    Hy = np.zeros((N, P, P))

    # Get initial fourier transform arrays
    prev_Ehat, prev_Xhat, prev_Yhat = myf.dSSt2(E[0]), myf.dSCt2(Hx[0]), myf.dCSt2(Hy[0])
    Ehat, Xhat, Yhat = np.zeros((P, P)), np.zeros((P, P)), np.zeros((P, P))
    for i in range(1, N):
        Ehat = ((1 - (parr * Dx) ** 2 - (qarr * Dy)** 2) * prev_Ehat \
        + 2 * qarr * Dy * prev_Xhat + 2 * parr * Dx * prev_Yhat + tau * Jhat[i - 1]) / (1 + (parr * Dx) ** 2 + (qarr * Dy)** 2)

        Xhat = prev_Xhat - qarr * Dy * (Ehat + prev_Ehat)

        Yhat = prev_Yhat - parr * Dx * (Ehat + prev_Ehat)


        E[i] = myf.idSSt2(Ehat)
        Hx[i] = myf.idSCt2(Xhat)
        Hy[i] = myf.idCSt2(Yhat)

    return E, Hx, Hy


if __name__ == '__main__':
    # part a 
    test_arr = np.array([[0,0,0],[0,1,0],[0,0,0]])
    print("Original array:")
    print(test_arr)
    print("Result of inverting sin-sin transform:")
    print(myf.idSSt2(myf.dSSt2(test_arr)))
    print("Result of inverting sin-cos transform:")
    print(myf.idSCt2(myf.dSCt2(test_arr)))
    print("Result of inverting cos-sin transform:")
    print(myf.idCSt2(myf.dCSt2(test_arr)))


    # part b
    E, Hx, Hy = Crank_Nicolson()

    # get indeces corresponding to the follow x and y vals
    xmid, ymid = 0.5, 0.5
    p, q = int(P * xmid // Lx),int(P * ymid // Ly)

    # get tracers
    E_tracer = E[:, p, q]
    Hx_tracer = Hx[:, p, 0]
    Hy_tracer = Hy[:, 0, q]

    plt.plot(time, E_tracer, label='$E_z(x=0.5, y=0.5)$')
    plt.plot(time, Hx_tracer, label='$H_x(x=0.5, y=0)$')
    plt.plot(time, Hy_tracer, label='$H_y(x=0, y=0.5)$')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('EM tracers (V/mc)')
    plt.title('EM tracers vs time')
    plt.savefig('q2b', dpi=300, bbox_inches='tight')
    plt.show()

    indeces = [42, 123] # indeces near max and min of E field
    
    for index in indeces:
        time = tau * index # get the time

        # transpose arrays for data to match properly
        plt.contourf(parr * Lx / P, qarr * Lx / P, np.transpose(E[index]), 60)
        plt.colorbar()
        plt.quiver(parr * Lx / P, qarr * Lx / P, np.transpose(Hx[index])/ c, -np.transpose(Hy[index])/c)
        plt.title("Electromagnetic field at time t={}s".format(time))
        plt.xlabel("$x/L_x$")
        plt.ylabel("$y/L_y$")
        plt.savefig("q2c_{}".format(index), dpi=300, bbox_inches='tight')
        plt.show()