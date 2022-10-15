import numpy as np
import matplotlib.pyplot as plt


# vec{r} = (x1_1, x1_2, )
def f(r): 
    #x(variable index)_(Nth particle)
    x1 = r[0]
    y1 = r[1]
    x2 = r[2]
    y2 = r[3]
    factor = 2/((x1-x2)**2+(y1-y2)**2)**7-1/((x1-x2)**2+(y1-y2)**2)**4
    fx1 = 12*(x1-x2)*factor
    fy1 = 12*(y1-y2)*factor
    fx2 = 12*(x2-x1)*factor
    fy2 = 12*(y2-y1)*factor
    return np.array([fx1, fy1, fx2, fy2])

def solve(r1, r2, v1, v2, dt=0.01):
    r = np.zeros((300, 4))
    r[0][0], r[0][1] = r1
    r[0][2], r[0][3] = r2
    v = np.zeros((300, 4))
    v[0][0], v[0][1] = v1
    v[0][2], v[0][3] = v2
    for i in range(0, len(r)-3): # dt -> 2dt
        if i == 0:
            v[1] = v[0]+dt/2*f(r[0])
        r[i+2] = r[i]+dt*v[i+1]
        k = dt*f(r[i+2])
        v[i+2] = v[i+1]+0.5*k
        v[i+3] = v[i+1]+k
    return r, v

if __name__ == '__main__':
    r1_sol, v1_sol = solve((4, 4), (5.2, 4), (0, 0), (0, 0))
    for i in range(0, len(r1_sol)):
        plt.plot(r1_sol[i][0], r1_sol[i][1], marker='o', linestyle='None', color='blue')
        plt.plot(r1_sol[i][2], r1_sol[i][3], marker='o', linestyle='None', color='red')
    plt.show()