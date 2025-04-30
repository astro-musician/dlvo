# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:52:29 2025

@author: veillon
"""

import numpy as np
import matplotlib.pyplot as plt

# Paramètres ajustés
epsilon_0 = 8.854e-12  # Permittivité du vide en F/m
epsilon = 78  # Permittivité relative de l'eau
Psi_0 = 0.017  # Potentiel de surface en V
A_H = 8.3e-21  # Constante de Hamaker en J
r_p = 950  # Rayon des particules en nm
I = 0.015
kappa = I**(1/2) / 0.304  # Longueur de Debye en 1/nm
h = np.linspace(0, 10, 1000)  # Plage de distances en m

# Potentiel DLVO (combinaison du terme électrostatique et du terme de London)
def potential_dlvo(h, r_p, epsilon_0, epsilon, Psi_0, A_H, kappa):
    # Terme électrostatique (répulsion) qui décroît exponentiellement
    electrostatic_term = 2 * np.pi * r_p *10**-9 * epsilon_0 * epsilon * Psi_0**2 * np.exp(-kappa * h)
    
    # Terme de London (attraction), avec un déclin rapide à grande distance
    london_term = -(A_H / 6) * (2 * r_p**2 / (h * (h + 4 * r_p)) + 
                               2 * r_p**2 / ((h + 2 * r_p)**2) + 
                               np.log((h * (h + 4 * r_p)) / ((h + 2 * r_p)**2)))
    

    # Retourne la somme des deux termes
    return (electrostatic_term + london_term)/(1.38e-23*300)

# Calcul du potentiel
V = potential_dlvo(h, r_p, epsilon_0, epsilon, Psi_0, A_H, kappa)

# Tracé du potentiel DLVO
plt.figure(figsize=(8, 6))
plt.plot(h, V, label="Potentiel DLVO")  # Conversion des unités pour affichage
plt.xlabel("Distance h (nm)")
plt.ylabel("V(h)/k_bT")
plt.title("Potentiel DLVO entre deux sphères (ajusté)")
plt.ylim(-200,100)
plt.grid(True)
plt.legend()
plt.show()

# F_dlvo = - 










