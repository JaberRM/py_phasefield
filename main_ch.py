# Simple Cahn-Hilliard phase-field model
import numpy as np
from matplotlib import pyplot as plt


def calc_del2(phi):
    # calculate the Laplacian of a scalar field with finite difference (periodic BC)
    dx = 1
    dy = 1
    phi_fx = np.roll(phi, 1, axis=0)    # forward in x
    phi_bx = np.roll(phi, -1, axis=0)   # backward in x
    phi_fy = np.roll(phi, 1, axis=1)    # forward in y
    phi_by = np.roll(phi, -1, axis=1)   # backward in y
    return (phi_fx + phi_fy + phi_bx + phi_by - 4 * phi) / (dx * dy)


def calc_force(phi):
    # Driving force for evolution of Phi consists of bulk + interface contributions
    kappa = 5.0e-1      # Sort of interface energy, larger value, more diffuse, more round shapes
    mobility = 5.0e-3   # mobility of the solutes (different than diffusivity but related!)

    del2phi = calc_del2(phi)
    # Gibbs free energy and its derivative, 4th order double-well shape is assumed
    # G = 5 * (phi - 0.5). ^ 4 - (phi - 0.5). ^ 2
    dGdphi = 20 * np.power((phi - 0.5), 3) - 2 * (phi - 0.5)

    # the chemical potential including the contribution from the interface
    mu = dGdphi - 2 * kappa * del2phi

    return mobility * calc_del2(mu)


def update_phi(phi):
    # Update the order parameter. This is an implicit, iterative update.
    dt = 1  # time step size, will diverge is not small enough

    f = calc_force(phi)
    phi_temp = phi + f * dt
    converged = 0
    for Iter in range(10):
        f_temp = calc_force(phi_temp)
        phi_try = phi + (f + f_temp) * dt / 2
        dphi = phi_try - phi_temp
        tol = np.max(np.abs(dphi[:]))
        if tol < 1.0e-6:
            phi = phi_try
            converged = 1
        else:
            phi_temp = phi_try
    if converged == 0:
        print("Iteration did not converge!")
        exit()
    return phi


# Main run part
# Initialization / geometry and parameters
Nx = 64        # system size in x
Ny = 64        # system size in y
# Phi is the scalar field (order parameter), here it is the concentration field
Phi = 0.5 + np.random.rand(Nx, Ny)/100

plt.ion()
plt.show()
frame = 0

for i in range(0, 10000):
    Phi = update_phi(Phi)
    frame = frame + 1

    # plotting and reporting every 10 steps, note that mean(Phi) is constant here, the system is conserved. Phi can be
    # solute concentration for example.
    if np.mod(frame, 10) == 1:
        print("Frame = ", frame, "/ Max Phi = ", np.max(Phi), "/ Min Phi =  ",
              np.min(Phi), "/ Mean Phi =", np.mean(Phi))
        # plotting part
        plt.clf()
        plt.imshow(Phi, interpolation='bilinear')
        plt.colorbar()
        plt.title("Frame = " + str(frame))
        plt.draw()
        # plt.clim(0, 1)
        plt.pause(0.00000001)