import numpy as np
import matplotlib.pyplot as plt
from time import time

stop = 100
t = np.zeros(stop)
N = np.zeros(stop)
for l in range(stop):
    n = l + 2
    A = np.ones([n, n], float)*3
    B = np.ones([n, n], float)*2
    C = np.zeros([n, n], float)

    start = time()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k]*B[k,j] 
    end = time()

    N[l] = n
    t[l] = end - start

plt.plot(N, t)
plt.show()