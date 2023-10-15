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
        self.E = np.zeros(self.mesh.nodes.shape)
        (self.mu_p, self.mu_n) = self.physics.computeCarrierMobilities(self.E)
        (self.D_p, self.D_n) = self.physics.computeCarrierDiffusivities(self.mu_p, self.mu_n)

        # Initialize properties
        self.psi = self.phi_p = self.phi_n = self.p = self.n = np.zeros(self.mesh.nodes.shape)
        self.boundary_values: Physics.BoundaryValues = None
        self.psi_R = self.psi_L = None

    def guessPotential(self) -> np.ndarray:
        return (self.psi_R - self.psi_L) / self.mesh.length * self.mesh.nodes + self.psi_L
    
    def initializeFEM(self, applied_potential: float):
        # Compute boundary values
        V = applied_potential
        self.boundary_values = self.physics.computeBoundaryValues(self.N, V)

        # Compute initial values
        self.psi_L = self.boundary_values.electric_potential[0]
        self.psi_R = self.boundary_values.electric_potential[1]
        self.psi = self.guessPotential()
        self.E = self.physics.computeElectricField(self.psi)
        (self.mu_p, self.mu_n) = self.physics.computeCarrierMobilities(self.E)
        (self.D_p, self.D_n) = self.physics.computeCarrierDiffusivities(self.mu_p, self.mu_n)

        (self.phi_p, self.phi_n) = self.physics.computeQuasiFermiPotentials(self.psi, self.D_p, self.D_n, self.boundary_values)
        (self.p, self.n) = self.physics.computeCarrierConcentrations(self.psi, self.phi_p, self.phi_n)

    def step(self, alpha: float) -> float:
        # Calculate electric potential update
        psi_update = self.physics.computeElectricPotential(self.N, self.p, self.n, [self.psi_L, self.psi_R])
        psi_new = (1 - alpha) * self.psi + alpha * psi_update

        # Update the mean error
        mean_error = np.mean(np.abs(self.psi - psi_new))

        # Update the electric potential
        self.psi = psi_new
        self.E = self.physics.computeElectricField(self.psi)
        (self.mu_p, self.mu_n) = self.physics.computeCarrierMobilities(self.E)
        (self.D_p, self.D_n) = self.physics.computeCarrierDiffusivities(self.mu_p, self.mu_n)

        # Recalculate the EQFP and HQFP
        u = self.physics.computeRecombinationRate(self.phi_p, self.phi_n, self.psi)
        (self.phi_p, self.phi_n) = self.physics.computeQuasiFermiPotentials(self.psi, self.D_p, self.D_n, self.boundary_values, u)

        # Recalculate the hole and electron concentrations
        (self.p, self.n) = self.physics.computeCarrierConcentrations(self.psi, self.phi_p, self.phi_n)

        return mean_error
    
    def solve(self, applied_potential: float, alpha: float, threshold: float):
        assert 1 - alpha >= 0

        self.initializeFEM(applied_potential)

        mean_error = threshold
        while mean_error >= threshold:
            mean_error = self.step(alpha)

    def plot(self):
        fig, ax = plt.subplots(3,1, figsize=(10,10))
        fig.tight_layout(pad=5)
        x = self.mesh.nodes * Material.L_D
        ax[0].plot(x, self.psi)
        ax[0].set_title("Electric Potential $(V_T)$")
        ax[0].set_xlabel("Position (cm)")
        ax[0].set_xlim([x[0],x[-1]])
        ax[1].plot(x, self.phi_n, label="$\phi_n$")
        ax[1].plot(x, self.phi_p, label="$\phi_p$")
        ax[1].set_title("Quasi-Fermi Potentials $(V_T)$")
        ax[1].set_xlabel("Position (cm)")
        ax[1].legend()
        ax[1].set_xlim([x[0],x[-1]])
        ax[2].plot(x, self.p * Material.n_i)
        ax[2].plot(x, self.n * Material.n_i)
        ax[2].set_title("Carrier Concentrations $(cm^{-3})$")
        ax[2].set_xlabel("Position (cm)")
        ax[2].set_xlim([x[0], x[-1]])
        plt.show()