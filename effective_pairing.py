# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:53:59 2024

@author: Gabriel
"""

import numpy as np
import matplotlib.pyplot as plt

from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x

def get_Hamiltonian(k_x, k_y, V_x, V_y, alpha, mu, Delta_0, gamma):
    """ Periodic Hamiltonian in x and y.
    """
    chi_k = gamma * (k_x**2 + k_y**2) - mu
    B_x_plus = -V_x - alpha*k_y
    B_x_minus = -V_x + alpha*k_y
    B_y_plus = -V_y + alpha*k_x
    B_y_minus = -V_y - alpha*k_x
    H = (chi_k * np.kron(tau_z, sigma_0) +
         B_x_plus * np.kron((tau_0 + tau_z)/2, sigma_x) +
         B_y_plus * np.kron((tau_0 + tau_z)/2, sigma_y) +
         B_x_minus * np.kron((tau_0 - tau_z)/2, sigma_x) +
         B_y_minus * np.kron((tau_0 - tau_z)/2, sigma_y) +
        + Delta_0*np.kron(tau_x, sigma_0)
            )
    return H


L_y = 100
k_x = 0
alpha = 0.56
mu = 0  #0
Delta_0 = 0.2
gamma = 1
V_x = 2*Delta_0
V_y = 0   #0
k_y_values = np.pi/L_y*np.arange(-L_y, L_y)

fig, ax = plt.subplots()
ax.set_xlabel(r"$k_y$")
ax.set_ylabel(r"$E(k_x=" + f"{k_x})$")

def get_energy(k_x, k_y, V_x, V_y, alpha, mu, Delta_0, gamma):
    energies = np.zeros(4)
    for i in range(4):
        H = get_Hamiltonian(k_x, k_y, V_x, V_y, alpha, mu, Delta_0, gamma)
        energies[i] = np.linalg.eigvalsh(H)[i]
    return energies

def get_energy_approximated(k_x, k_y, V_x, alpha, mu, Delta_0, gamma):
    chi_k = gamma * (k_x**2 + k_y**2) - mu
    Delta_x = Delta_0 * alpha / V_x
    B_abs_plus_k = np.sqrt( (V_x - alpha*k_y)**2 + (alpha*k_x)**2 )
    B_abs_minus_k = np.sqrt( (V_x + alpha*k_y)**2 + (alpha*k_x)**2 )
    a = 1
    b = B_abs_plus_k - B_abs_minus_k
    c = ( chi_k*(B_abs_plus_k + B_abs_minus_k) - B_abs_plus_k*B_abs_minus_k
         - (Delta_x*k_x)**2 - chi_k**2 )
    discriminant = ( (B_abs_plus_k + B_abs_minus_k) - 2*chi_k )**2 + 4*(Delta_x*k_x)
    # E_plus = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    # E_minus = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    E_plus = (-b + np.sqrt(discriminant)) / (2*a)
    E_minus = (-b - np.sqrt(discriminant)) / (2*a)
    return np.array([E_plus, E_minus])

def discriminant(k_x, k_y, V_x, alpha, mu, Delta_0, gamma):
    chi_k = gamma * (k_x**2 + k_y**2) - mu
    Delta_x = Delta_0 * alpha / V_x
    B_abs_plus_k = np.sqrt( (V_x - alpha*k_y)**2 + (alpha*k_x)**2 )
    B_abs_minus_k = np.sqrt( (V_x + alpha*k_y)**2 + (alpha*k_x)**2 )
    discriminant = ( (B_abs_plus_k + B_abs_minus_k) - 2*chi_k )**2 + 4*(Delta_x*k_x)
    return discriminant

ax.set_title(r"$k_x=$" + f"{k_x}"+
         r"; $\alpha=$" + f"{alpha}"+
         r"; $\mu=$" + f"{mu}"+
         r"; $\Delta_0=$" + f"{Delta_0}"+
         r"; $V_x=$" + f"{np.round(V_x, 2)}")

ax.plot(k_y_values, [get_energy(k_x, k_y, V_x, V_y, alpha, mu, Delta_0, gamma)[0]
                     for k_y in k_y_values], "--k", label=r"$E$", alpha=0.5)
ax.plot(k_y_values, [get_energy(k_x, k_y, V_x, V_y, alpha, mu, Delta_0, gamma)[1]
                     for k_y in k_y_values], "--k", label=r"$E$", alpha=0.5)
ax.plot(k_y_values, [get_energy(k_x, k_y, V_x, V_y, alpha, mu, Delta_0, gamma)[2]
                     for k_y in k_y_values], "--k", label=r"$E$", alpha=0.5)
ax.plot(k_y_values, [get_energy(k_x, k_y, V_x, V_y, alpha, mu, Delta_0, gamma)[3]
                     for k_y in k_y_values], "--k", label=r"$E$", alpha=0.5)

ax.plot(k_y_values, [get_energy_approximated(k_x, k_y, V_x, alpha, mu, Delta_0, gamma)
                     for k_y in k_y_values], "--b", label=r"$E_{eff}$ approximation",
        alpha=0.5)

k_plus = np.sqrt((mu + V_x)/gamma)
k_minus = -np.sqrt((mu + V_x)/gamma)

# ax.plot([k_plus, k_minus], [get_energy_approximated(k_x, k, V_x, alpha, mu, Delta_0, gamma)
#                             for k in [k_plus, k_minus]], "o")


# ax.plot([k_plus, k_minus], [0, 0], "o")

def get_energy_Carlos(k_x, k_y, V_x, alpha, mu, Delta_0, gamma):
    chi_k = gamma * (k_x**2 + k_y**2) - mu
    E_k = np.sqrt(chi_k**2 + Delta_0**2)
    E_plus = alpha*k_y*chi_k/E_k + np.sqrt( (E_k-V_x)**2 + (alpha*k_x*Delta_0/E_k)**2)
    E_minus = alpha*k_y*chi_k/E_k - np.sqrt( (E_k-V_x)**2 + (alpha*k_x*Delta_0/E_k)**2)
    return np.array([E_plus, E_minus])

<<<<<<< HEAD
# ax.plot(k_y_values, [get_energy_Carlos(k_x, k_y, V_x, alpha, mu, Delta_0, gamma)
#                      for k_y in k_y_values], "r", label=r"$E_{eff}$ (Carlos)")
=======
ax.plot(k_y_values, [get_energy_Carlos(k_x, k_y, V_x, alpha, mu, Delta_0, gamma)
                     for k_y in k_y_values], "r", label=r"$E_{eff}$ (Carlos)",
        alpha=0.5)
>>>>>>> fda8dd1f616b72e7a8b6f404cb8ab849c46ecaa3


# ax.plot([k_plus, k_minus], [get_energy_Carlos(k_x, k, V_x, alpha, mu, Delta_0, gamma)
#                             for k in [k_plus, k_minus]], "o")

# ax.legend()

ax.plot(k_y_values, np.zeros_like(k_y_values))
ax.set_ylim((-1, 1))
plt.tight_layout()
plt.show()

#%% 3D plot
from matplotlib import cm
L_x = 10
L_y = 10

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.pi/L_x*np.arange(-L_x, L_x)
Y = np.pi/L_y*np.arange(-L_y, L_y)
X, Y = np.meshgrid(X, Y)
Z_1, Z_2 = get_energy_approximated(X, Y, V_x, alpha, mu, Delta_0, gamma)

surface_1 = ax.plot_surface(X, Y, Z_1,
                          cmap=cm.coolwarm, antialiased=False)
surface_2 = ax.plot_surface(X, Y, Z_2,
                          cmap=cm.coolwarm, antialiased=False)

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
# ax.set_zlim(-1/2, 1/2)

#%% Contour plot
L_x = 1000
L_y = 1000

fig, ax = plt.subplots()
X = np.pi/L_x*np.arange(-L_x, L_x)
Y = np.pi/L_y*np.arange(-L_y, L_y)
X, Y = np.meshgrid(X, Y)
Z_1, Z_2 = get_energy_approximated(X, Y, V_x, alpha, mu, Delta_0, gamma)

surface_1 = ax.contour(X, Y, Z_1, levels=np.linspace(-1, 1, 5), colors="blue")
surface_2 = ax.contour(X, Y, Z_2, levels=np.linspace(-1, 1, 5), colors="red")
# surface_3 = ax.contour(X, Y, Z_1 - Z_2, levels=[0.01], colors="black")
ax.plot(0, k_plus, "ok")
ax.plot(0, k_minus, "ok")

ax.clabel(surface_1, surface_1.levels)
ax.clabel(surface_2, surface_2.levels)

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
# ax.set_zlim(-1/2, 1/2)
