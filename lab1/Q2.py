import numpy as np
import matplotlib.pyplot as plt
import myFunctions as myf

# initial conditions for earth
xe0, ye0 = 1, 0  # AU
vex0, vey0 = 0, 6.18  # AU / yr

# initial conditions for asteroid
xa0, ya0 = 3.3, 0  # AU
vax0, vay0 = 0, 3.46  # AU / yr

def main(mjup, x0, y0, vx0, vy0, timespan, ver):
    """
    Main algorithm for computing and plotting orbits
    INPUT: 
    mjup is the mass of jupiter
    (x0, y0, vx0, vy0) are the initial conditions for the small body
    timespan is the number of Earth years that are simulater
    ver is a dictionary which specifies what the small body and the large body are
    OUTPUT: Plots
    """
    # initialize time step and end variables 
    dt = 10 ** -3 # yr
    N = int(timespan/dt) # steps in 10 earth years

    # Initialize arrays for small body with initial conditions
    x = np.zeros(N)
    y = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    x[0] = x0
    y[0] = y0
    vx[0] = vx0 
    vy[0] = vy0 

    # initial conditions for jupiter
    xj0, yj0 = 5.2, 0  # AU
    vjx0, vjy0 = 0, 2.63  # AU / yr

    # Initialize arrays for earth with initial conditions
    xj = np.zeros(N)
    yj = np.zeros(N)
    vjx = np.zeros(N)
    vjy = np.zeros(N)
    xj[0] = xj0
    yj[0] = yj0
    vjx[0] = vjx0 
    vjy[0] = vjy0 

    # create array to determine net gravitational force on earth due to sun and jupiter
    # g = np.zeros(N)

    #Loop for 10**4 steps and solve equations using Euler-Cromer method
    for i in range(1, N):
        # update conditions for jupiter
        vjx[i] = vjx[i-1] + myf.gravity(xj[i-1], yj[i-1])[0]*dt
        vjy[i] = vjy[i-1] + myf.gravity(xj[i-1], yj[i-1])[1]*dt
        xj[i] = xj[i-1] + vjx[i]*dt
        yj[i] = yj[i-1] + vjy[i]*dt

        

        # get the net acceleration of the small body
        g = myf.gravity(x[i-1], y[i-1]) + mjup * myf.gravity(x[i-1] - xj[i-1], y[i-1] - yj[i-1])

        # update conditions for the small body
        vx[i] = vx[i-1] + g[0]*dt
        vy[i] = vy[i-1] + g[1]*dt
        x[i] = x[i-1] + vx[i]*dt
        y[i] = y[i-1] + vy[i]*dt

    #Plot position of planet (x, y) space

    plt.gcf().set_size_inches(6,6)
    plt.plot(x, y, label=ver['small'])
    plt.plot(0, 0, label='Sun', marker='o', linestyle='none')
    plt.plot(xj, yj, label=ver['big'])
    plt.legend(loc='upper right')
    plt.xlabel('$x$ (AU)')
    plt.ylabel('$y$ (AU)')
    plt.title('Orbital Position')
    plt.savefig('lab1/img/Q2position_{0}_{1}'.format(ver['small'], ver['big']))
    plt.clf()

if __name__ == '__main__':
    # part a
    main(myf.MJUP, xe0, ye0, vex0, vey0, 10, {'small': 'Earth', 'big': 'Jupiter'})

    # part b
    main(1000 * myf.MJUP, xe0, ye0, vex0, vey0, 3, {'small': 'Earth', 'big': '1000xJupiter'})

    # part c
    main(myf.MJUP, xa0, ya0, vax0, vay0, 20, {'small': 'Asteroid', 'big': 'Jupiter'})