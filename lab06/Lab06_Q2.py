import numpy as np
import matplotlib.pyplot as plt
import Lab06_MyFunctions as myf

N = 16
Lx = 4.0
Ly = 4.0
dx = Lx/np.sqrt(N)
dy = Ly/np.sqrt(N)
x_grid = np.arange(dx/2, Lx, dx)
y_grid = np.arange(dy/2, Ly, dy)
xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
x_initial = xx_grid.flatten()
y_initial = yy_grid.flatten()

r0 = np.array([x_initial, y_initial]).transpose().flatten()

v0 = np.zeros(2*N)

r, v, energy = myf.solve_N(r0, v0, 1000)
DOFs = len(r)
fig, ax = plt.subplots()

# create a way to cycle through colours
color = iter(plt.cm.rainbow(np.linspace(0, 1, DOFs // 2)))
for i in range(0, DOFs, 2):
    # get x and y position data for the particle
    x, y = r[i:i+2]
    num_points = len(x)

    c= next(color)  # iterate to next colour
    # plot each data point with increasing opacity
    for j in range(num_points):
        plt.plot(x[j], y[j], marker='.', linestyle='None', 
        markersize=2, color=c, alpha=j/num_points)
plt.show()

# plot energy
plt.plot(energy)
avg = np.mean(energy)
plt.plot(avg * np.ones(len(energy)), linestyle='--')
plt.plot(1.01 * avg * np.ones(len(energy)), linestyle='--')
plt.plot(0.99 * avg * np.ones(len(energy)), linestyle='--')
plt.show()