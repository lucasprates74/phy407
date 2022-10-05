# SolveLinear.py
# Python module for PHY407
# Paul Kushner, 2015-09-26
# Modifications by Nicolas Grisouard, 2018-09-26
# This module contains useful routines for solving linear systems of equations.
# Based on gausselim.py from Newman
from numpy import empty
import numpy as np
# The following will be useful for partial pivoting
from numpy import empty, copy


def GaussElim(A_in, v_in):
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to
    temporary variables
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)

    for m in range(N):
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x


def PartialPivot(A_in, v_in):
    """ In this function, code the partial pivot (see Newman p. 222) """
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)

    for m in range(N):
        # Pivot rows  
        initial_div = A[m, m] 
        index = m
        for j in range(m, N): # Loop through the N-m elements in the mth col
            if abs(A[j, m]) > initial_div: 
                index = j # Index to swap rows with
                initial_div = abs(A[j, m]) # New maximum to beat
        A[m, :], A[index, :] = copy(A[index, :]), copy(A[m, :]) # flip matrix rows
        # Swap vector rows
        temp = v[m]
        v[m] = v[index]
        v[index] = temp
        # Proceed with regular Gauss elim alg
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x

if __name__ == "__main__": # For the printout check
    A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], dtype=float)
    v = np.array([-4, 3, 9, 7], dtype=float)
    print('Using A and v from (6.16) from the textbook')
    expected = np.linalg.solve(A, v) 
    print('Expected from numpy: w = {0}, x = {1}, y = {2}, z = {3}'.format(expected[0], expected[1], expected[2], expected[3]))
    partial_answer = PartialPivot(A, v)
    print('From Partial Pivot: w = {0}, x = {1}, y = {2}, z = {3}'.format(expected[0], expected[1], expected[2], expected[3]))