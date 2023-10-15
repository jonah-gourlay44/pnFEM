import numpy as np
from mesh import Mesh
from constants import *
from typing import List

class Physics:

    class BoundaryValues:

        def __init__(self):

            self.hole_quasi_fermi_potential = []
            self.electron_quasi_fermi_potential = []
            self.electric_potential = []

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def computeShapeFunctionDerivateTerms(self):
        a = np.zeros((2, 2))
        dN = 1. / self.mesh.dl
        a[0] = dN * np.array([ 1, -1])
        a[1] = dN * np.array([-1,  1])

        return a
    
    def computeShapeFunctionValueTerms(self):
        return np.array([[self.mesh.dl, 0],[0, self.mesh.dl]])

    def computeDopingProfile(self):
        return np.where(self.mesh.nodes < self.mesh.mid_point, Material.N_D, -Material.N_A) / Material.n_i
    
    def computeCarrierMobilities(self, E: np.ndarray):
        mu_n = Material.mu_0n / (np.sqrt(1.0 + Material.N_D / (Material.N_CD + Material.N_D / (Material.A_n) ** 2)) + E * Material.mu_0n / Material.v_scat_n)
        mu_p = Material.mu_0p / (np.sqrt(1.0 + Material.N_A / (Material.N_CA + Material.N_A / (Material.A_p) ** 2)) + E * Material.mu_0p / Material.v_scat_p)

        return (mu_p, mu_n)

    def computeCarrierDiffusivities(self, hole_mobility: np.ndarray, electron_mobility: np.ndarray):
        mobility = np.vectorize(lambda mu : mu * k_B * T / q)

        D_p = mobility(hole_mobility)
        D_n = mobility(electron_mobility)

        return (D_p, D_n)

    def computeBoundaryValues(self, doping_profile: np.ndarray, applied_potential: float):
        N = doping_profile
        V = applied_potential

        phi_p_L = phi_n_L = 0
        phi_p_R = phi_n_R = V * q / (k_B * T)
        psi_L = phi_p_L - np.log(np.sqrt((N[0]/2.)**2 + 1) - N[0]/2.)
        psi_R = phi_p_R - np.log(np.sqrt((N[-1]/2.)**2 + 1) - N[-1]/2.)

        values = self.BoundaryValues()
        values.hole_quasi_fermi_potential = [phi_p_L, phi_p_R]
        values.electron_quasi_fermi_potential = [phi_n_L, phi_n_R]
        values.electric_potential = [psi_L, psi_R]

        return values
    
    def computeElectricPotential(self, doping_profile: np.ndarray, hole_concentration: np.ndarray, electron_concentration: np.ndarray, boundary_values: List[float]):
        A = np.zeros((self.mesh.num_nodes, self.mesh.num_nodes))
        b = np.zeros((self.mesh.num_nodes, 1))

        N = doping_profile
        p = hole_concentration
        n = electron_concentration

        psi_L = boundary_values[0]
        psi_R = boundary_values[1]

        # Fill matrices
        for i in range(0, self.mesh.num_edges):
            nds = self.mesh.edges[i]

            dNi_dNj = self.computeShapeFunctionDerivateTerms()
            Ni = self.computeShapeFunctionValueTerms()

            a = dNi_dNj
            _b = np.zeros((2,1))
            _b[0] = np.trapz(Ni[0]*(N[nds] + p[nds] - n[nds]), dx=self.mesh.dl)
            _b[1] = np.trapz(Ni[1]*(N[nds] + p[nds] - n[nds]), dx=self.mesh.dl)

            A[np.ix_(nds, nds)] += a
            b[nds] += _b

        # Enforce boundary conditions
        A[0][0] = 1; A[0][1:] = 0
        A[-1][-1] = 1; A[-1][:-1] = 0

        b[0] = psi_L; b[-1] = psi_R

        # Solve set of linear equations
        return np.linalg.solve(A, b).reshape(self.mesh.nodes.shape)
    
    def computeElectricField(self, electric_potential: np.ndarray):
        psi = electric_potential

        return np.gradient(-psi, self.mesh.dl)
    
    def computeRelativeReciprocalDiffusivities(self, hole_diffusivity: np.ndarray, electron_diffusivity: np.ndarray):
        D_p = hole_diffusivity
        D_n = electron_diffusivity

        gamma_p = Material.D_0p / D_p
        gamma_n = Material.D_0n / D_n

        return (gamma_p, gamma_n)
    
    def computeCarrierConcentrations(self, electric_potential: np.ndarray, hole_quasi_fermi_potential: np.ndarray, electron_quasi_fermi_potential: np.ndarray):
        psi = electric_potential
        phi_n = electron_quasi_fermi_potential
        phi_p = hole_quasi_fermi_potential
        
        p = np.exp(phi_p - psi)
        n = np.exp(psi - phi_n)

        return (p, n)
    
    def computeRecombinationRate(self, hole_quasi_fermi_potential: np.ndarray, electron_quasi_fermi_potential: np.ndarray, electric_potential: np.ndarray):
        phi_p = hole_quasi_fermi_potential
        phi_n = electron_quasi_fermi_potential
        psi = electric_potential
        
        numerator = Material.n_i * (np.exp(phi_p - phi_n) - 1)
        denominator = Material.tau_p * (np.exp(psi - phi_n) + np.exp(Material.phi_t)) + Material.tau_n * (np.exp(phi_p - psi) + np.exp(-Material.phi_t))

        return numerator / denominator
    
    def computeRecombinationCurrent(self, recombination_rate: np.ndarray):
        u = recombination_rate
    
        indices = np.linspace(0, self.mesh.nodes.size - 1, self.mesh.nodes.size, dtype=int)
        constant = Material.L_D**2 / (Material.D_0 * Material.n_i)

        J_r_vec = np.vectorize(lambda i : np.trapz(u[:i+1], self.mesh.nodes[:i+1]))
        J_r = J_r_vec(indices)

        return constant * J_r


    def computeElectronQuasiFermiPotential(self, electric_potential: np.ndarray, relative_reciprocal_diffusivity: np.ndarray, boundary_values: List[float], recombination_current: np.ndarray = None):
        gamma = relative_reciprocal_diffusivity
        J_r = recombination_current
        phi_n_L = boundary_values[0]
        phi_n_R = boundary_values[1]
        psi = electric_potential

        indices = np.linspace(0, self.mesh.nodes.size - 1, self.mesh.nodes.size, dtype=int)

        # Calculate functions F and FR
        psi_exp = np.exp(-psi)
        F_integrand = gamma * psi_exp
        F_func = np.vectorize(lambda i : np.trapz(F_integrand[i:], self.mesh.nodes[i:]))
        F = F_func(indices)

        FR = np.zeros(F.shape)
        if J_r is not None:
            FR_integrand = J_r * gamma * psi_exp
            FR_func = np.vectorize(lambda i : np.trapz(FR_integrand[i:], self.mesh.nodes[i:]))
            FR = FR_func(indices)

        # Calculate constant part of electron current density
        J_n_const = (np.exp(-phi_n_R) - np.exp(-phi_n_L) - FR[0]) / F[0]

        # Calculate phi_n
        phi_n = -np.log(-J_n_const * F - FR + np.exp(-phi_n_R))

        return phi_n
    
    def computeHoleQuasiFermiPotential(self, electric_potential: np.ndarray, relative_reciprocal_diffusivity: np.ndarray, boundary_values: List[float], recombination_current: np.ndarray = None):
        gamma = relative_reciprocal_diffusivity
        J_r = recombination_current
        phi_p_L = boundary_values[0]
        phi_p_R = boundary_values[1]
        psi = electric_potential

        indices = np.linspace(0, self.mesh.nodes.size - 1, self.mesh.nodes.size, dtype=int)

        # Calculate functions F and FR
        psi_exp = np.exp(psi)
        F_integrand = gamma * psi_exp
        F_func = np.vectorize(lambda i : np.trapz(F_integrand[i:], self.mesh.nodes[i:]))
        F = F_func(indices)

        FR = np.zeros(F.shape)
        if J_r is not None:
            FR_integrand = J_r * gamma * psi_exp
            FR_func = np.vectorize(lambda i : np.trapz(FR_integrand[i:], self.mesh.nodes[i:]))
            FR = FR_func(indices)

        # Calculate constant part of hole current density
        J_p_const = (np.exp(phi_p_L) - np.exp(phi_p_R) + FR[0]) / F[0]

        # Calculate hole quasi fermi potential
        phi_p = np.log(J_p_const * F - FR + np.exp(phi_p_R))
        
        return phi_p

    def computeQuasiFermiPotentials(self, electric_potential: np.ndarray, hole_diffusivity: np.ndarray, electron_diffusivity: np.ndarray, boundary_values: BoundaryValues, recombination_rate: np.ndarray = None):
        u = recombination_rate
        psi = electric_potential
        gamma_p = hole_diffusivity
        gamma_n = electron_diffusivity

        J_r = None
        if u is not None:
            J_r = self.computeRecombinationCurrent(u)
        phi_p = self.computeHoleQuasiFermiPotential(psi, gamma_p, boundary_values.hole_quasi_fermi_potential, J_r)
        phi_n = self.computeElectronQuasiFermiPotential(psi, gamma_n, boundary_values.electron_quasi_fermi_potential, J_r)

        return (phi_p, phi_n)