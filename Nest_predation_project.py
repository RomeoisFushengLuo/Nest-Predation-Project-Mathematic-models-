#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:30:50 2023

@author: fushengluo (Group4)
"""


import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve, approx_fprime
import matplotlib.pyplot as plt
import sympy as sp

#### The belowing is the 3-d plot of the model PEB with egg-tossing

# Parameter values
k = 0.3
mu = 0.02
c = 0.5
theta = 2
lmbda = 100
g = 0.015
phi = 0.3
gamma = 0.2

# Define the system of equations
def system_equations(variables):
    P, E, B = variables
    eq1 = mu * B * P - k * P
    eq2 = c * B * E * (1 - theta * E / (1 + E**2)) - lmbda * E * B / (1 + E) - g * E * B
    eq3 = lmbda * E * B / (1 + E) - phi * B * P - gamma * B
    return [eq1, eq2, eq3]

# Find equilibrium points
initial_guess = [400, 300, 200]  # Initial guess for P, E, B
equilibrium_points = fsolve(system_equations, initial_guess)

print("Equilibrium Points:")
print("P:", equilibrium_points[0])
print("E:", equilibrium_points[1])
print("B:", equilibrium_points[2])

# Calculate Jacobian matrix at equilibrium points using approx_fprime
epsilon = 1e-8  # Small perturbation

equilibrium_Jacobian = []
for i in range(len(equilibrium_points)):
    perturbation = np.zeros_like(equilibrium_points)
    perturbation[i] = epsilon

    def objective_function(vars):
        return system_equations(vars)[i]

    partial_derivative = approx_fprime(equilibrium_points, objective_function, perturbation)
    equilibrium_Jacobian.append(partial_derivative)

# Reshape the Jacobian matrix to a 3x3 matrix
equilibrium_Jacobian = np.array(equilibrium_Jacobian).reshape((3, 3))

print("\nJacobian Matrix at Equilibrium Points:")
print(equilibrium_Jacobian)

# Define the system of differential equations
def system(y, t):
    P, E, B = y
    dydt = [
        mu * B * P - k * P,
        c * B * E * (1 - theta * E / (1 + E**2)) - lmbda * E * B / (1 + E) - g * E * B,
        lmbda * E * B / (1 + E) - phi * B * P - gamma * B
    ]
    return dydt

# Set time points
t = np.linspace(0, 100, 1000)

# Adjusted perturbation
perturbation = np.abs(np.random.normal(scale=10, size=3))

# Solve the system
sol = odeint(system, equilibrium_points + perturbation, t)

# Plot the trajectory in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory with a solid line
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label='Trajectory', linewidth=2)

# Scatter plot for equilibrium point and initial value
ax.scatter(*equilibrium_points, color='red', s=100, label='Equilibrium Point')
ax.scatter(*(equilibrium_points + perturbation), color='green', s=100, label='Initial Value')

# Annotate equilibrium points
for i, txt in enumerate(['P', 'E', 'B']):
    ax.text(equilibrium_points[0], equilibrium_points[1], equilibrium_points[2], txt, fontsize=12, color='red')

# Annotate initial points
for i, txt in enumerate(['P', 'E', 'B']):
    ax.text(equilibrium_points[0] + perturbation[0],
            equilibrium_points[1] + perturbation[1],
            equilibrium_points[2] + perturbation[2], f'{txt}_0', fontsize=12, color='green')

# Add labels and title
ax.set_xlabel('P', fontsize=14)
ax.set_ylabel('E', fontsize=14)
ax.set_zlabel('B', fontsize=14)
ax.set_title('Phase portrait of the nonzero equilibria plus small perturbation (with egg tossing)', fontsize=16)

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Customize legend
ax.legend(fontsize=12)

# Show the improved plot
plt.show()



#### The belowing is the 3-d plot of the model PEB without egg-tossing

# Parameter values
k = 0.3
mu = 0.02
c = 0.5
lmbda = 100
phi = 0.3
gamma = 0.2

# Define the system of equations
def system_equations(variables):
    P, E, B = variables
    eq1 = mu * B * P - k * P
    eq2 = c * E * B - lmbda * E * B / (1 + E)
    eq3 = lmbda * E * B / (1 + E) - phi * B * P - gamma * B
    return np.array([eq1, eq2, eq3])

# Find equilibrium points
initial_guess = [400, 300, 200]  # Initial guess for P, E, B
equilibrium_points = fsolve(system_equations, initial_guess)

print("Equilibrium Points:")
print("P:", equilibrium_points[0])
print("E:", equilibrium_points[1])
print("B:", equilibrium_points[2])

# Calculate Jacobian matrix at equilibrium points using approx_fprime
epsilon = 1e-8  # Small perturbation

equilibrium_Jacobian = []
for i in range(len(equilibrium_points)):
    perturbation = np.zeros_like(equilibrium_points)
    perturbation[i] = epsilon

    def objective_function(vars):
        return system_equations(vars)[i]

    partial_derivative = approx_fprime(equilibrium_points, objective_function, perturbation)
    equilibrium_Jacobian.append(partial_derivative)

# Reshape the Jacobian matrix to a 3x3 matrix
equilibrium_Jacobian = np.array(equilibrium_Jacobian).reshape((3, 3))

print("\nJacobian Matrix at Equilibrium Points:")
print(equilibrium_Jacobian)

# Define the system of differential equations
def system(y, t):
    P, E, B = y
    dydt = [
        mu * B * P - k * P,
        c * E * B - (lmbda * E * B) / (1 + E),
        (lmbda * E * B) / (1 + E) - phi * B * P - gamma * B
    ]
    return dydt

# Set time points
t = np.linspace(0, 100, 1000)

# Adjusted perturbation
perturbation = np.abs(np.random.normal(scale=10, size=3))

# Solve the system
sol = odeint(system, equilibrium_points + perturbation, t)

# Plot the trajectory in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory with a solid line
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label='Trajectory', linewidth=2)

# Scatter plot for equilibrium point and initial value
ax.scatter(*equilibrium_points, color='red', s=100, label='Equilibrium Point')
ax.scatter(*(equilibrium_points + perturbation), color='green', s=100, label='Initial Value')

# Annotate equilibrium points
for i, txt in enumerate(['P', 'E', 'B']):
    ax.text(equilibrium_points[0], equilibrium_points[1], equilibrium_points[2], txt, fontsize=12, color='red')

# Annotate initial points
for i, txt in enumerate(['P', 'E', 'B']):
    ax.text(equilibrium_points[0] + perturbation[0],
            equilibrium_points[1] + perturbation[1],
            equilibrium_points[2] + perturbation[2], f'{txt}_0', fontsize=12, color='green')

# Add labels and title
ax.set_xlabel('P', fontsize=14)
ax.set_ylabel('E', fontsize=14)
ax.set_zlabel('B', fontsize=14)
ax.set_title('Phase portrait of the nonzero equilibria plus small perturbation (without egg tossing)', fontsize=16)

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Customize legend
ax.legend(fontsize=12)

# Show the improved plot
plt.show()





#### The belowing is the 2-d plot between birds v.s. predators in the PEB model without egg-tossing

# Parameter values
k = 0.3
mu = 0.02
c = 0.5
lmbda = 100
phi = 0.3
gamma = 0.2

# Equilibrium points
equilibrium_points = [331, 207, 15]

# System of differential equations
def system(y, t):
    P, E, B = y
    dydt = [
        mu * B * P - k * P,
        c * E * B - (lmbda * E * B) / (1 + E),
        (lmbda * E * B) / (1 + E) - phi * B * P - gamma * B
    ]
    return dydt

# Time points
t = np.linspace(0, 100, 1000)

# Solve the system
sol = odeint(system, equilibrium_points, t)

# Plotting
plt.figure(figsize=(10, 6))

# Plot for P vs t
plt.subplot(3, 1, 1)
plt.plot(t, sol[:, 0], label='P')
plt.xlabel('Time (t)')
plt.ylabel('Predator')
plt.title('Predators vs Time(without egg tossing)')
plt.legend()



# Plot for B vs t
plt.subplot(3, 1, 3)
plt.plot(t, sol[:, 2], label='B', color='green')
plt.xlabel('Time (t)')
plt.ylabel('')
plt.title('Birds vs Time(without egg tossing)')
plt.legend()



plt.tight_layout()
plt.show()




#### The belowing is the 2-d plot between birds v.s. predators in the PEB model with egg-tossing

# Parameter values
k = 0.3
mu = 0.02
c = 0.5
theta = 2
lmbda = 100
g = 0.015
phi = 0.3
gamma = 0.2

# Equilibrium points
equilibrium_points = [340, 220, 30]

# System of differential equations
def system(y, t):
    P, E, B = y
    dydt = [
        mu * B * P - k * P,
        c * B * E * (1 - theta * E / (1 + E**2)) - lmbda * E * B / (1 + E) - g * E * B,
        lmbda * E * B / (1 + E) - phi * B * P - gamma * B
    ]
    return dydt

# Time points
t = np.linspace(0, 100, 1000)

# Solve the system
sol = odeint(system, equilibrium_points, t)

# Plotting
plt.figure(figsize=(10, 6))

# Plot for P vs t
plt.subplot(3, 1, 1)
plt.plot(t, sol[:, 0], label='P')
plt.xlabel('Time (t)')
plt.ylabel('P')
plt.title('Predators vs Time(with egg tossing)')
plt.legend()



# Plot for B vs t
plt.subplot(3, 1, 3)
plt.plot(t, sol[:, 2], label='B', color='green')
plt.xlabel('Time (t)')
plt.ylabel('B')
plt.title('Birds vs Time (with egg tossing)')
plt.legend()

plt.tight_layout()
plt.show()


#### The belowing is the 2-d plot of birds / predators v.s. time in the model without egg-tossing

# Parameter values
k = 0.3
mu = 0.02
c = 0.5
lmbda = 100
phi = 0.3
gamma = 0.2

# Equilibrium points
equilibrium_points = [331, 200, 15]

# System of differential equations
def system(y, t):
    P, E, B = y
    dydt = [
        mu * B * P - k * P,
        c * E * B - lmbda * E * B / (1 + E),
        lmbda * E * B / (1 + E) - phi * B * P - gamma * B
    ]
    return dydt

# Time points
t = np.linspace(0, 100, 1000)

# Solve the system
sol = odeint(system, equilibrium_points, t)

# Plotting
plt.figure(figsize=(15, 5))

# Plot for P vs B
plt.subplot(1, 3, 2)
plt.plot(sol[:, 0], sol[:, 2], label='P vs B', color='green')
plt.xlabel('Number of Predator (units)')
plt.ylabel('Number of Bird (units)')
plt.title('Predators vs Birds (without egg tossing)')
plt.legend()

plt.tight_layout()
plt.show()


#### The belowing is the 2-d plot of birds / predators v.s. time in the model with egg-tossing

# Parameter values
k = 0.3
mu = 0.02
c = 0.5
theta = 2
lmbda = 100
g = 0.015
phi = 0.3
gamma = 0.2

# Equilibrium points
equilibrium_points = [340, 220, 30]

# System of differential equations
def system(y, t):
    P, E, B = y
    dydt = [
        mu * B * P - k * P,
        c * B * E * (1 - theta * E / (1 + E**2)) - lmbda * E * B / (1 + E) - g * E * B,
        lmbda * E * B / (1 + E) - phi * B * P - gamma * B
    ]
    return dydt

# Time points
t = np.linspace(0, 100, 1000)

# Solve the system
sol = odeint(system, equilibrium_points, t)

# Plotting
plt.figure(figsize=(15, 5))


# Plot for P vs B
plt.subplot(1, 3, 2)
plt.plot(sol[:, 0], sol[:, 2], label='P vs B', color='green')
plt.xlabel('Number of Predator (units)')
plt.ylabel('Number of Bird (units)')
plt.title('Predators vs Birds (with egg tossing)')
plt.legend()



plt.tight_layout()
plt.show()


#### The belowing is the model to compute the real part of the complex root in a cubic polynomials
### In PEB model without egg-tossing

# Define the symbolic variables
P, E, lambd, theta, phi, c, x = sp.symbols('P E lambda theta phi c x')

# Define the cubic equation
equation = ((1+E**2)**2)*x**3 - (-35)*E*((2*theta-lambd)*c*E**2 + (1-c)*lambd)* x**2 + P*(-35)*0.02*phi*((1+E**2)**2)*x - P*(-35)*0.02*phi*(-35)*E*((2*theta-lambd)*c*E**2 + (1-c)*lambd)

# Solve for the roots
roots = sp.solve(equation, x)

# Extract the real part of the complex root
real_part = sp.re(roots[0])

# Print the result
print("Real part of the complex root:", real_part)
