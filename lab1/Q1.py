from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
import myFunctions as myf
# initial conditions
x0, y0 = .47, 0  # AU
vx0, vy0 = 0, 8.17  # AU / yr
dt = 10 ** -4 # yr
N = int(1/dt) #steps

#Initialize arrays with initial conditions
x = np.zeros(N)
y = np.zeros(N)
vx = np.zeros(N)
vy = np.zeros(N)
x[0] = x0
y[0] = y0
vx[0] = vx0 
vy[0] = vy0 

#Loop for 10**3 steps and solve equations using Euler-Cromer method
for i in range(1, N):
    vx[i] = vx[i-1] + myf.gravity(x[i-1], y[i-1])[0]*dt
    vy[i] = vy[i-1] + myf.gravity(x[i-1], y[i-1])[1]*dt
    x[i] = x[i-1] + vx[i]*dt
    y[i] = y[i-1] + vy[i]*dt

#Compute angular momentum
L = myf.MJUP*(x*vy - y*vx) #CHANGE MASS TO MERCURY

#Plot position of planet (x, y) space
time = np.array(range(N))*dt
plt.plot(x, y)
plt.xlabel('$x$ (AU)')
plt.ylabel('$y$ (AU)')
plt.show()

#Plot velocity components
plt.plot(time, vx, label='$v_x$')
plt.plot(time, vy, label='$v_y$')
plt.legend()
plt.xlabel('Time $t$ (yr)')
plt.ylabel('Velocity (m/s)')
plt.show()

#Plot angular momentum magnitude
plt.plot(time, L)
plt.xlabel('Time $t$ (yr)')
plt.ylabel('Angular Momentum $||\\overrightarrow{{L}}||$ ()')
plt.show()
