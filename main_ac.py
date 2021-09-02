# Simple Allen-Cahn phase-field model
import numpy as np
from matplotlib import pyplot as plt


def calc_del2(phi):
    # Laplacian of a 2D data with finite difference (periodic BC)
    dx = 1
    dy = 1
    phi_fx = np.roll(phi, 1, axis=0)
    phi_bx = np.roll(phi, -1, axis=0)
    phi_fy = np.roll(phi, 1, axis=1)
    phi_by = np.roll(phi, -1, axis=1)
    return (phi_fx + phi_fy + phi_bx + phi_by - 4 * phi) / (dx * dy)


def calc_dfdphi(phi, temperature):
    # Derivative of the Gibbs energy with respect to the order parameter
    # Temperature in range of 100 to 1000
    G1 = 1 * (temperature / 1000)
    G2 = 1 - G1
    c = 50
    d = 0
    e = G1
    a = -2*c - 3*(G2-G1-c)
    b = G2-G1-c-a
    # f(Phi) = a*Phi**4 + b*Phi**3 + c*Phi**2 + d*x + e
    return 4 * a * phi ** 3 + 3 * b * phi ** 2 + 2 * c * phi + d


def calc_force(phi, temperature):
    # Driving force for evolution of Phi consists of interface  + bulk contributions
    kappa = 30  # Sort of interface energy, larger value, more diffuse, more round shapes
    mobility = 2.0e-4
    return mobility * (2 * kappa * calc_del2(phi) - calc_dfdphi(phi, temperature))


def update_phi(phi, temperature, nsteps):
    dt = 1
    dphi = 0
    for Step in range(0, nsteps):
        dphi = dt * calc_force(phi, temperature)
        phi = phi + dphi
    return phi, dphi


# Main run part
# Initialization / geometry and parameters
Nx = 64        # system size in x
Ny = 64        # system size in y
NSteps = 50    # how many steps to update the phi in each call

# Phi is the scalar field (order parameter), 0 means the first phase, 1 means the second phase
Phi = np.random.rand(Nx, Ny)

# Some fun tests here
# phase 1 (Phi=0) is more stable at low temperature, phase 2 (Phi=1) is more stable at high temperatures.
# This is set in the form of the energy. At 500K, both phases have the same energy and evolution is controlled
# based on the initial random distribution and interface energy
# Temperature of system (things work and converge for values between 100 and 1000)
Temperature = 500

plt.ion()
plt.show()
frame = 0

for i in range(0, 400):
    [Phi, dPhi] = update_phi(Phi, Temperature, NSteps)
    frame = frame + 1

    print("Frame = ", frame, "/ volume fraction = ", np.mean(Phi), "/ dPhi = ", np.max(np.abs(dPhi)))
    # plotting part
    plt.clf()
    plt.imshow(Phi, interpolation='bilinear')
    plt.colorbar()
    plt.title("Frame = " + str(frame))
    plt.draw()
    plt.clim(0, 1)
    plt.pause(0.00001)

