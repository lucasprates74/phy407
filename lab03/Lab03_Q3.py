import numpy as np 
import matplotlib.pyplot as plt
import Lab03_myFunctions as myf

N = 100

def H(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return 2*x*H(n-1, x)-2*(n-1)*H(n-2, x)

def psi(n, x):
    coeff = H(n, x)/np.sqrt(np.sqrt(np.pi)*float(2**n)*float(np.math.factorial(n)))
    return coeff*np.exp(-x**2/2)

def psi_x(n, x):
    if n == 0:
        return -x*np.exp(-x**2/2)/np.pi**0.25
    coeff = (-x*H(n, x)+2*n*H(n-1, x))/np.sqrt(np.sqrt(np.pi)*float(2**n)*float(np.math.factorial(n)))
    return coeff*np.exp(-x**2/2)

def expec_x_squared_integrand(n, x):
    return np.tan(x)**2/np.cos(x)**2*abs(psi(n, np.tan(x)))**2

def expec_p_squared_integrand(n, x):
    return abs(psi_x(n, np.tan(x)))**2/np.cos(x)**2

def expec_x_squared(n):
    x, w = myf.gaussxwab(N, -np.pi/2, np.pi/2)
    s = 0
    for i in range(len(x)):
        s += w[i]*expec_x_squared_integrand(n, x[i])
    return s

def expec_p_squared(n):
    x, w = myf.gaussxwab(N, -np.pi/2, np.pi/2)
    s = 0
    for i in range(len(x)):
        s += w[i]*expec_p_squared_integrand(n, x[i])
    return s

def total_energy(n):
    return 0.5*(expec_x_squared(n)+expec_p_squared(n))

def x_uncert(n):
    return np.sqrt(expec_x_squared(n))

def p_uncert(n):
    return np.sqrt(expec_p_squared(n))

if __name__ == '__main__':
   #part a)
    x = np.arange(-4, 4, 0.1)
    for i in range(4):
        plt.plot(x, psi(i, x), label='$\\psi_{{{0}}}(x)$'.format(i))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('$\\psi_n(x)$')
    plt.show()

    #part b)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, psi(30, x), label='$\\psi_{{{0}}}(x)$'.format(30))
    plt.xlabel('x')
    plt.ylabel('$\\psi_30(x)$')
    plt.show() 
    
    #part c)
    x = np.arange(0, 15, 1)
    for i in x:
        plt.plot(i, x_uncert(i), color='blue', marker='x', ms=10)
        plt.plot(i, p_uncert(i), color='red', marker='o')
        plt.plot(i, x_uncert(i)*p_uncert(i), color='green', marker='s')
    plt.plot([], [], color='blue', marker='x', label='$\\Delta X$')
    plt.plot([], [], color='red', marker='o', label='$\\Delta P$')
    plt.plot([], [], color='green', marker='s', label='$\\Delta X\\Delta P$')
    plt.plot(x, x+0.5, color='orange', label='$\\left(n+\\frac{{1}}{{2}}\\right)$')
    plt.xlabel('Energy level $n$')
    plt.ylabel('Uncertainty')
    plt.legend()
    plt.show()

    for i in x:
        plt.plot(i, total_energy(i), marker='o', color='blue')
    plt.plot([], [], color='blue', marker='o', label='$E_{{n}}$')
    plt.plot(x, x+0.5, color='orange', label='$\\left(n+\\frac{{1}}{{2}}\\right)$')
    plt.xlabel('Energy level $n$')
    plt.ylabel('Total energy $E_n$')
    plt.legend()
    plt.show()
    

