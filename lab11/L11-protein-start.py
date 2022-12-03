"""
Strter code for protein folding
Author: Nicolas Grisuard, based on a script by Paul Kushner
"""

from random import random, randrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def calc_energy(monomer_coords, monomer_array, N, eps):
    """ Compute energy of tertiary structure of protein """
    energy = 0.0

    # compute energy due to all adjacencies (incl. directly bonded monomers)
    for i in range(N):
        for nghbr in [[-1, 0], [1, 0], [0, -1], [0, 1]]:  # 4 neighbours
            nghbr_monomer = monomer_array[monomer_coords[i, 0] + nghbr[0],
                                          monomer_coords[i, 1]+nghbr[1]]

            if nghbr_monomer == 1:  # check neighbour is not empty
                energy += eps

    # divide by 2 to correct for double-counting
    energy = .5*energy

    # correct energy to not count directly bonded monomer neighbours
    energy -= (N-1)*eps

    return energy


def dist(position1, position2):
    """ Compute distance """
    return ((position1[0]-position2[0])**2+(position1[1]-position2[1])**2)**.5


font = {'family': 'DejaVu Sans', 'size': 14}  # adjust fonts
rc('font', **font)
dpi = 150


def get_figures(eps = -5.0, N = 30, n = int(1e5), T = 1.5, T_i = 1.5, dT=0):
    """
    Generates the figures showing the final protein structure and plots the energy 
    as function of Monte-Carlo step. 
    INPUT:
    eps, the interaction energy
    N, the protein length
    T, Monte Carlo Temperature 
    Ti, the initial monte carlo temperature
    Tsteps, the number of decreasing temperature steps. If Tsteps=1, temperature is constant
    n, number of Monte-Carlo steps
    """

    # create the temperature array
    if dT == 0 :
        T_array = np.zeros(n) + T
    else: 
        T_f = T
        T_steps = int((T_i - T_f) / dT + 1)
        T_array = np.zeros(n)
        for step in range(T_steps):
            T_array[step*n//T_steps:(step+1)*n//T_steps] = (T_i-T_f)*(1-step/(T_steps-1)) + T_f


    energy_array = np.zeros(n)  # initialize array to hold energy
    print(T_array[0], T_array[-1])
    # initialize arrays to store protein information
    # 1st column is x coordinates, 2nd column is y coordinates, of all N monomers
    monomer_coords = np.zeros((N, 2), dtype='int')

    # initialize position of polymer as horizontal line in middle of domain
    monomer_coords[:, 0] = range(N//2, 3*N//2)
    monomer_coords[:, 1] = N

    # 2D array representing lattice,
    # equal to 0 when a lattice point is empty,
    # and equal to 1 when there is a monomer at the lattice point
    monomer_array = np.zeros((2*N+1, 2*N+1), dtype='int')

    # fill lattice array
    for i in range(N):
        monomer_array[monomer_coords[i, 0], monomer_coords[i, 1]] = 1

    # calculate energy of initial protein structure
    energy = calc_energy(monomer_coords, monomer_array, N, eps)


    dirs_to_neighs = {0:  np.array([-1, -1]), 1: np.array([-1, 1]), 2: np.array([1, 1]),3: np.array([1, -1])}
    # do Monte Carlo procedure to find optimal protein structure
    for j in range(n):
        energy_array[j] = energy

        # move protein back to centre of array
        shift_x = int(np.mean(monomer_coords[:, 0])-N)
        shift_y = int(np.mean(monomer_coords[:, 1])-N)
        monomer_coords[:, 0] -= shift_x
        monomer_coords[:, 1] -= shift_y
        monomer_array = np.roll(monomer_array, -shift_x, axis=0)
        monomer_array = np.roll(monomer_array, -shift_y, axis=1)

        # pick random monomer
        i = randrange(N)
        cur_monomer_pos = monomer_coords[i, :]

        # pick random diagonal neighbour for monomer
        neighbour = dirs_to_neighs[randrange(4)]

        new_monomer_pos = cur_monomer_pos + neighbour

        # check if neighbour lattice point is empty
        if monomer_array[new_monomer_pos[0], new_monomer_pos[1]] == 0:
            # check if it is possible to move monomer to new position without
            # stretching chain
            distance_okay = False
            if i == 0:
                distance_okay = (dist(new_monomer_pos, monomer_coords[i+1, :]) < 1.1)
            elif i == N-1:
                distance_okay = (dist(new_monomer_pos, monomer_coords[i-1, :]) < 1.1)
            else:
                distance_okay = (
                    dist(new_monomer_pos, monomer_coords[i-1, :]) < 1.1 
                    and dist(new_monomer_pos, monomer_coords[i+1, :]) < 1.1
                )

            if distance_okay:
                # calculate new energy
                new_monomer_coords = np.copy(monomer_coords)
                new_monomer_coords[i, :] = new_monomer_pos

                new_monomer_array = np.copy(monomer_array)
                new_monomer_array[cur_monomer_pos[0], cur_monomer_pos[1]] = 0
                new_monomer_array[new_monomer_pos[0], new_monomer_pos[1]] = 1

                new_energy = calc_energy(new_monomer_coords, new_monomer_array, N, eps)

                if random() < np.exp(-(new_energy-energy)/T_array[j]):
                    # make switch
                    energy = new_energy
                    monomer_coords = np.copy(new_monomer_coords)
                    monomer_array = np.copy(new_monomer_array)

    plt.figure()
    plt.title('$T$ = {0:.1f}, $N$ = {1:d}'.format(T, N))
    plt.plot(energy_array)
    plt.xlabel('MC step')
    plt.ylabel('Energy')
    plt.grid()
    plt.tight_layout()
    plt.savefig('energy_vs_step_T{0:d}_N{1:d}_n{2:d}.png'.format(int(10*T), N, n),
                dpi=dpi)

    plt.figure()
    plt.plot(monomer_coords[:, 0], monomer_coords[:, 1], '-k')  # plot bonds
    plt.title('$T$ = {0:.1f}, Energy = {1:.1f}'.format(T, energy))
    # plot monomers
    for i in range(N):
        plt.plot(monomer_coords[i, 0], monomer_coords[i, 1], '.r', markersize=15)
    plt.xlim([N/3.0, 5.0*N/3.0])
    plt.ylim([N/3.0, 5.0*N/3.0])
    plt.axis('equal')
    # plt.xticks([])  # we just want to see the shape
    # plt.yticks([])
    plt.tight_layout()
    plt.savefig('final_protein_T{0:d}_N{1:d}_n{2:d}.png'.format(int(10*T), N, n),
                dpi=dpi)

    print('Energy averaged over last quarter of simulations is: {0:.2f}'
        .format(np.mean(energy_array[3*n//4:])))
    print('Energy averaged over last half of simulations is: {0:.2f}'
        .format(np.mean(energy_array[n//2:])))
    plt.show()

    if dT != 0:
        # create arrays for temperature, mean energy, and std of energy
        T_return = np.zeros(T_steps)
        mean_E = np.zeros(T_steps)
        std_E = np.zeros(T_steps)
        for step in range(T_steps):
            T_return[step] = T_array[step*n//T_steps]
            mean_E[step] = np.mean(energy_array[step*n//T_steps:(step+1)*n//T_steps])
            std_E[step] = np.std(energy_array[step*n//T_steps:(step+1)*n//T_steps])

        plt.figure()
        plt.errorbar(T_return, mean_E, std_E, marker='o', linestyle='none')  # plot bonds
        plt.title('Energy vs Temperature')
        plt.xlabel('$T$')
        plt.ylabel('$E$')
        plt.tight_layout()
        if n !=10**7:
            plt.savefig('EnergyVTemp')
        plt.show()


if __name__ == '__main__':

    # 2a
    get_figures()
    get_figures(T=0.5)
    get_figures(T=5)

    # # 2b
    get_figures(T=0.5, n=10**6)
    get_figures(T=5, n=10**6)

    # # 2d
    get_figures(T=0.5, n=2 * 10**6, T_i=3.5, dT=1)

    # 2e
    get_figures(T=0.5, n= 10**7, T_i=10, dT=0.5)
    

