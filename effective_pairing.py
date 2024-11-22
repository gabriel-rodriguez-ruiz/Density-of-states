# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:53:59 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt

def get_energy(k_x, k_y, V_x, alpha, mu, Delta_0):
    chi_k = (k_x**2 + k_y**2) - mu
    r = np.sqrt((V_x - alpha*k_y)**2 + (alpha*k_x)**2)
    E = np.sqrt( (chi_k - r)**2 + (Delta_0*alpha*k_x/V_x)**2 )
    return np.array([E, -E])

L_y = 500
k_x = 0
V_x = 6
alpha = 0.56
mu = 2
Delta_0 = 1
k_y_values = 2*np.pi/L_y*np.arange(-L_y, L_y)

fig, ax = plt.subplots()
ax.plot(k_y_values, [get_energy(k_x, k_y, V_x, alpha, mu, Delta_0)
                     for k_y in k_y_values], "b", label=r"$E_{eff}$")
ax.set_xlabel(r"$k_y$")
ax.set_ylabel(r"$E(k_x=0)$")

def get_energy_approximated(k_x, k_y, V_x, alpha, mu, Delta_0):
    chi_k = (k_x**2 + k_y**2) - mu
    E = V_x - alpha*k_y - chi_k
    return np.array([E, -E])

ax.set_title(r"$k_x=$" + f"{k_x}"+
         r"; $\alpha=$" + f"{alpha}"+
         r"; $\mu=$" + f"{mu}"+
         r"; $\Delta_0=$" + f"{Delta_0}"+
         r"; $V_x=$" + f"{V_x}")
# ax.plot(k_y_values, [get_energy_approximated(k_x, k_y, V_x, alpha, mu, Delta_0)
#                      for k_y in k_y_values], "--", label=r"$E_{eff}$ approximation")

def get_energy_Carlos(k_x, k_y, V_x, alpha, mu, Delta_0):
    chi_k = (k_x**2 + k_y**2) - mu
    E_k = np.sqrt(chi_k**2 + Delta_0**2)
    E_plus = alpha*k_y + np.sqrt( (E_k-V_x)**2 + (alpha*k_x*Delta_0/E_k)**2)
    E_minus = alpha*k_y - np.sqrt( (E_k-V_x)**2 + (alpha*k_x*Delta_0/E_k)**2)
    return np.array([E_plus, E_minus])

ax.plot(k_y_values, [get_energy_Carlos(k_x, k_y, V_x, alpha, mu, Delta_0)
                     for k_y in k_y_values], "--r", label=r"$E_{eff}$ (Carlos)")
ax.legend()
plt.tight_layout()
plt.show()