import numpy as np
import matplotlib.pyplot as plt

N = 4000
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
    r = np.zeros((N, 4))
    r[0][0], r[0][1] = r1
    r[0][2], r[0][3] = r2
    v = np.zeros((N, 4))
    v[0][0], v[0][1] = v1
    v[0][2], v[0][3] = v2
    v_half = np.zeros((N, 4))
    v_half[0] = v[0]+dt/2*f(r[0])
    for i in range(len(r)-1): # dt -> 2dt
        r[i+1] = r[i]+dt*v_half[i]
        k = dt*f(r[i+1])
        v[i+1] = v_half[i]+0.5*k
        v_half[i+1] = v_half[i]+k
    return r, v

if __name__ == '__main__':
    r1_sol, v1_sol = solve((4, 4), (5.2, 4), (0, 0), (0, 0))
    for i in range(0, len(r1_sol)):
        plt.plot(r1_sol[i][0], r1_sol[i][1], marker='.', linestyle='None', color='blue', alpha=i/N)
        plt.plot(r1_sol[i][2], r1_sol[i][3], marker='.', linestyle='None', color='red', alpha=i/N)
    plt.show()

    plt.plot(np.arange(0, N, 1), np.transpose(r1_sol)[0], color='blue')
    plt.plot(np.arange(0, N, 1), np.transpose(r1_sol)[2], color='red')
    plt.show()

    r1_sol, v1_sol = solve((4.5, 4), (5.2, 4), (0, 0), (0, 0))
    for i in range(0, len(r1_sol)):
        plt.plot(r1_sol[i][0], r1_sol[i][1], marker='.', linestyle='None', color='blue', alpha=i/N)
        plt.plot(r1_sol[i][2], r1_sol[i][3], marker='.', linestyle='None', color='red', alpha=i/N)
    plt.show()

    plt.plot(np.arange(0, N, 1), np.transpose(r1_sol)[0], color='blue')
    plt.plot(np.arange(0, N, 1), np.transpose(r1_sol)[2], color='red')
    plt.show()

    r1_sol, v1_sol = solve((2, 3), (3.5, 4.4), (0, 0), (0, 0))
    for i in range(0, len(r1_sol)):
        plt.plot(r1_sol[i][0], r1_sol[i][1], marker='.', linestyle='None', color='blue', alpha=i/N)
        plt.plot(r1_sol[i][2], r1_sol[i][3], marker='.', linestyle='None', color='red', alpha=i/N)
    plt.show()

    plt.plot(np.arange(0, N, 1), np.transpose(r1_sol)[0], color='blue')
    plt.plot(np.arange(0, N, 1), np.transpose(r1_sol)[2], color='red')
    plt.show()