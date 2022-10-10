import numpy as np

def EulerCromer(x0, v0, func, tstop, dt):
    """
    Solves the ode system 

    xdot = v
    vdot = func(x, v)

    given initial conditions x0, v0 for the time range 0 <= t <= end_time using
    the Euler Cromer methond with num_points samples.
    """

    t = np.arange(0, tstop, dt)
    num_points = len(t)
    print(num_points)
    x, v = np.zeros(num_points), np.zeros(num_points)

    # set initial conditions
    x[0], v[0] = x0, v0

    for i in range(1, num_points):
        v[i] = func(x[i - 1], v[i - 1]) * dt + v[i - 1]
        x[i] = v[i] * dt + x[i - 1]

    return t, x, v

from numpy import ones,copy,cos,tan,pi,linspace

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