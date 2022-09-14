"""
Q1 code. External function and variable calls to myFunction.py for gravity formulas and constants.
Authors: Sam De Abreu
"""
import numpy as np
import matplotlib.pyplot as plt
import myFunctions as myf
# initial conditions
x0, y0 = .47, 0  # AU
vx0, vy0 = 0, 8.17  # AU / yr
dt = 10  ** -4 # yr
N = int(1/dt) #steps

def main(func, ver): #Main algorithm for computing and plotting orbits
    """
    Main algorithm for computing and plotting orbits
    INPUT: func is the acceleration formula and ver is either Newtonian or Relativistic
    OUTPUT: Plots
    """
    #Initialize arrays with initial conditions
    x = np.zeros(N)
    y = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    x[0] = x0
    y[0] = y0
    vx[0] = vx0
    vy[0] = vy0
    #Loop for 10**4 steps and solve equations using Euler-Cromer method
    for i in range(1, N):
        vx[i] = vx[i-1] + func(x[i-1], y[i-1])[0]*dt
        vy[i] = vy[i-1] + func(x[i-1], y[i-1])[1]*dt
        x[i] = x[i-1] + vx[i]*dt
        y[i] = y[i-1] + vy[i]*dt

    #Compute angular momentum
    L = (x*vy - y*vx)

    #Plot position of planet (x, y) space
    time = np.array(range(N))*dt

    plt.gcf().set_size_inches(6,6)
    plt.plot(x, y)
    plt.xlabel('$x$ (AU)')
    plt.ylabel('$y$ (AU)')
    plt.title('Orbital Position ({})'.format(ver))
    plt.savefig('lab1/img/Q1position_{0}.png'.format(ver))
    plt.clf()

    #Plot velocity components
    plt.plot(time, vx, label='$v_x$')
    plt.plot(time, vy, label='$v_y$')
    plt.legend()
    plt.xlabel('Time $t$ (yr)')
    plt.ylabel('Velocity AU/yr)')
    plt.title('Velocity vs Time ({})'.format(ver))
    plt.savefig('lab1/img/Q1velocity_{0}.png'.format(ver))
    plt.clf()

    #Plot angular momentum magnitude
    print(L)
    plt.plot(time, L)
    plt.ylim(2,5)
    plt.xlabel('Time $t$ (yr)')
    plt.ylabel('Angular Momentum per unit mass $||\\overrightarrow{{L}}||/M_M$ (AU$^2$/yr)')
    plt.title('Angular Momentum vs Time ({})'.format(ver))
    plt.savefig('lab1/img/Q1angularmomentum_{0}.png'.format(ver))
    plt.clf()

if __name__ == '__main__':
    main(myf.gravity, 'Newtonian') #Newtonian orbits, Q1c
    main(myf.gravity_rel, 'Relativistic') #Relativistic orbits, Q1d
