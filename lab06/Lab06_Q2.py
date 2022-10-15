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
print(x_initial)
print(y_initial)
r0 = np.array([x_initial, y_initial]).transpose().flatten()
print(r0)
v0 = np.zeros(2*N)

r, v, energy = myf.solve_N(r0, v0, 1000)
DOFs = len(r)
for i in range(0, DOFs, 2):
    x, y = r[i:i+2]
    plt.plot(x,y, marker='.', linestyle='None')
plt.show()

plt.plot(energy)
avg = np.mean(energy)
plt.plot(avg * np.ones(len(energy)), linestyle='--')
plt.plot(1.01 * avg * np.ones(len(energy)), linestyle='--')
plt.plot(0.99 * avg * np.ones(len(energy)), linestyle='--')
plt.show()