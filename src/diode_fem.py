from mesh import Mesh
from physics import Physics
import numpy as np
from constants import *

import matplotlib.pyplot as plt

class DiodeFEM:

    def __init__(self, mesh: Mesh):
        self.physics = Physics(mesh)
        self.mesh = mesh
        self.N = self.physics.computeDopingProfile()
        (self.mu_p, self.mu_n) = self.physics.computeCarrierMobilities()
        (self.D_p, self.D_n) = self.physics.computeCarrierDiffusivities(self.mu_p, self.mu_n)

        # Initialize properties
        self.psi = self.phi_p = self.phi_n = self.p = self.n = np.zeros(self.mesh.nodes.shape)
    
    def solve(self, applied_potential: float, alpha: float, threshold: float):
        assert 1 - alpha >= 0

        # Compute boundary values
        V = applied_potential
        boundary_values = self.physics.computeBoundaryValues(self.N, V)

        # Compute initial values
        psi_L = boundary_values.electric_potential[0]
        psi_R = boundary_values.electric_potential[1]
        self.psi = (psi_R - psi_L)/(self.mesh.length)*self.mesh.nodes + psi_L

        (self.phi_p, self.phi_n) = self.physics.computeQuasiFermiPotentials(self.psi, np.zeros(self.psi.shape), self.D_p, self.D_p, boundary_values, False)
        (self.p, self.n) = self.physics.computeCarrierConcentrations(self.psi, self.phi_p, self.phi_n)

        mean_error = threshold
        while mean_error >= threshold:
            
            # Calculate electric potential update
            psi_update = self.physics.computeElectricPotential(self.N, self.p, self.n, [psi_L, psi_R])
            psi_new = (1 - alpha) * self.psi + alpha * psi_update

            # Update the mean error
            mean_error = np.mean(np.abs(self.psi - psi_new))

            # Update the electric potential
            self.psi = psi_new

            # Recalculate the EQFP and HQFP
            u = self.physics.computeRecombinationRate(self.phi_p, self.phi_n, self.psi)
            (self.phi_p, self.phi_n) = self.physics.computeQuasiFermiPotentials(self.psi, u, self.D_p, self.D_n, boundary_values, True)

            # Recalculate the hole and electron concentrations
            (self.p, self.n) = self.physics.computeCarrierConcentrations(self.psi, self.phi_p, self.phi_n)

    def plot(self):
        fig, ax = plt.subplots(3,1)
        ax[0].plot(self.psi)
        ax[0].set_ylim([-15,15])
        ax[1].plot(self.phi_n)
        ax[1].plot(self.phi_p)
        ax[1].set_ylim([-10,10])
        ax[2].plot(self.p)
        ax[2].plot(self.n)
        #ax[2].set_ylim([-1, 1e10])
        plt.show()