# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:26:26 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from parallelization import get_Energy_without_SOC

L_x = 400
L_y = 400
w_0 = 10
Delta = 0.2 # 0.2 ###############Normal state
mu = -39#2*(20*Delta-2*w_0)
theta = np.pi/2
Lambda = 0.56      #0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
h = 1e-2        #1e-2
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
n_cores = 8
params = {"L_x": L_x, "L_y": L_y, "w_0": w_0,
          "mu": mu, "Delta": Delta, "theta": theta,
           "Lambda": Lambda,
          "h": h , "k_x_values": k_x_values,
          "k_y_values": k_y_values, "h": h,
          "Lambda": Lambda}
phi_x_values = np.array([-h, 0, h])
phi_y_values = np.array([-h, 0, h])
B_x = 2
B_y = 2

E = get_Energy_without_SOC(k_x_values, [0], [0], [0], w_0, mu,
               Delta, B_x, B_y)
fig, ax = plt.subplots()
ax.plot(k_x_values, E[:,0,0,0,0])
ax.plot(k_x_values, E[:,0,0,0,1])
ax.plot(k_x_values, E[:,0,0,0,2])
ax.plot(k_x_values, E[:,0,0,0,3])

plt.show()

#%%
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda):
    """ Periodic Hamiltonian in x and y with flux.
    """
    H = (
        -2*w_0*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
        + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
                    + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
                    - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
                    - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
        + Delta*np.kron(tau_x, sigma_0)
            ) * 1/2
    return H

def get_Energy_2(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda):
    energies = np.zeros((len(k_x_values), len(k_y_values),
                        len(phi_x_values), len(phi_y_values), 4))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for k, phi_x in enumerate(phi_x_values):
                for l, phi_y in enumerate(phi_y_values):
                    for m in range(4):
                        H = get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda)
                        energies[i, j, k, l, m] = np.linalg.eigvalsh(H)[m]
    return energies

E_2 = get_Energy_2(k_x_values, [0], phi_x_values, phi_y_values, w_0, mu,
               Delta, B_x, B_y, Lambda)
fig, ax = plt.subplots()
ax.plot(k_x_values, E_2[:,0,0,0,0])
ax.plot(k_x_values, E_2[:,0,0,0,1])
ax.plot(k_x_values, E_2[:,0,0,0,2])
ax.plot(k_x_values, E_2[:,0,0,0,3])

plt.show()

#%% Analytic solution
from analytic_energy import GetAnalyticEnergies

E_3 = GetAnalyticEnergies(k_x_values, np.array([0]), phi_x_values, phi_y_values, w_0, mu,
               Delta, B_x, B_y, Lambda)
fig, ax = plt.subplots()
ax.plot(k_x_values, E_3[:,0,0,0,0])
ax.plot(k_x_values, E_3[:,0,0,0,1])
ax.plot(k_x_values, E_3[:,0,0,0,2])
ax.plot(k_x_values, E_3[:,0,0,0,3])

plt.show()
