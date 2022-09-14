"""
To test the runtime of matrix multipilcation using for loops (O(N^3)) or np.dot (~O(1)). Results are then plotted.
Authors: Sam De Abreu & Lucas Prates
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time

#Initialize variables 
skip = 4 #Only compute for every 4th step
N = 200 #Size of NxN matrix 
t = [] #Array to hold all times
t_np = [] #Array to hold all numpy.dot times

#Compute times for all NxN matrices using both methods
for n in range(0, N, skip): 
    A = np.ones([n, n], float)*3
    B = np.ones([n, n], float)*2
    C = np.zeros([n, n], float) #Product matrix of AxB

    #Normal method
    start = time()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k]*B[k,j] #Compute C = AxB
    end = time()
    t.append(end - start)

    #numpy method
    start = time()
    C = np.dot(A, B)
    end = time()
    t_np.append(end - start)

#Plot t(N)
input = list(range(N//skip))
plt.plot(input, t)
plt.xlabel('Input Size $N$')
plt.ylabel('Time to Run $t$ (s)')
plt.title('Matrix Multipilcation Runtime $t(N)$')
plt.savefig('lab1/img/Q3Fig1.png')
plt.clf()

#Plot t(N^3)
input = np.array(list(range(N//skip)))**3
plt.plot(input, t)
plt.xlabel('Input Size Cubed $N^3$')
plt.ylabel('Time to Run $t$ (s)')
plt.title('Matrix Multiplication Runtime $t(N^3)$')
plt.savefig('lab1/img/Q3Fig2.png')
plt.clf()

#Plot t(N) for numpy method
input = np.array(list(range(N//skip)))
plt.plot(input, t_np)
plt.xlabel('Input Size $N$')
plt.ylabel('Time to Run $t$ (s)')
plt.title('Matrix Multiplication Runtime $t(N)$ using np.dot')
plt.ylim(-0.1, 0.1)
plt.savefig('lab1/img/Q3Fig3.png')
plt.clf()