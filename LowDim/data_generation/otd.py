#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of calculating the Optimally Time-Dependent (OTD) modes for a 
low-dimensional dynamical system to demonstrate the transient growth 
over different dirrections.

"""

import numpy as np
from scipy.integrate import odeint
np.random.seed(0)

# Define the system and its parameters
def low_dimensional_2(state, t, a1, a2, epsilon, b):
    z1, z2, z3 = state
    dz1_dt = -a1 * z1  + epsilon * z2 + b * z3 
    dz2_dt = epsilon**-1 * z1 - a2 * z2 
    dz3_dt = b * z3 * (1 / np.sqrt(z1**2 + z2**2) - 1)
    return [dz1_dt, dz2_dt, dz3_dt]




def solution_low_dimensional(initial_state, t_eval, a1, a2, epsilon, b):
    # Solve the system of differential equations
    solution = odeint(low_dimensional_2, initial_state, t_eval, args=(a1, a2, epsilon, b))

    # Extract solutions
    z1 = solution[:, 0]
    z2 = solution[:, 1]
    z3 = solution[:, 2]
    
    return z1, z2, z3


# Compute the Linearized System (Jacobian)
def jacobian(state, a1, a2, epsilon, b):
    z1, z2, z3 = state
    return np.array([
        [-a1, epsilon, b],
        [epsilon**-1, -a2, 0],
        [-b * z3 * z1 / (z1**2 + z2**2)**(3/2), -b * z3 * z2 / (z1**2 + z2**2)**(3/2), b * (1 / np.sqrt(z1**2 + z2**2) - 1)]
    ])



# Initialize the OTD Modes
def initialize_otd_modes(n, r):
    Q = np.zeros((n, r))
    Q[:, 0] = np.array([1, 0, 0])  # Set the first mode to [1, 0, 0]
    if r > 1:
        Q[:, 1:], _ = np.linalg.qr(np.random.randn(n, r - 1))
        Q, _ = np.linalg.qr(Q)
    return Q


# Evolve the OTD Modes using the OTD evolution equation
def evolve_otd_modes(U, jacobian, state, dt):
    J = jacobian(state, a1, a2, epsilon, b)
    dU_dt = J @ U - U @ (U.T @ J @ U)
    U += dt * dU_dt
    # Re-orthonormalize the modes
    U = U / np.linalg.norm(U, axis=0)
    return U



def solution_otd_modes(t_eval, z1, z2, z3):
    U = initialize_otd_modes(n, r)

    dt = t_eval[1] - t_eval[0]
    otd_modes = [U]
    
    for i in range(1, len(t_eval)):
        state = np.array([z1[i-1], z2[i-1], z3[i-1]])
        tmp = jacobian(state, a1, a2, epsilon, b) @ U - U @ (U.T @ jacobian(state, a1, a2, epsilon, b) @ U)
        U = U + tmp * dt
        U = U / np.linalg.norm(U, axis=0)
        otd_modes.append(U)

    otd_modes = np.array(otd_modes)
    
    return otd_modes

