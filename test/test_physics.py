import pytest
import numpy as np
import numpy.testing as testing

from mesh import Mesh
from physics import Physics
from constants import Material

@pytest.fixture
def physics_fixture():
    mesh = Mesh(10, 0.02)
    return Physics(mesh)

def test_material_values(physics_fixture: Physics):
    physics = physics_fixture
    
    electric_field = np.zeros(physics.mesh.nodes.shape)

    # Mobility
    (mu_p, mu_n) = physics.computeCarrierMobilities(electric_field)

    # Sanity check
    testing.assert_array_less(mu_p, mu_n)
    
    mu_n_max = np.ones(mu_n.shape) * 1500. # cm^2/(V*s)
    mu_n_min = np.ones(mu_n.shape) * 800.

    testing.assert_array_less(mu_n, mu_n_max)
    testing.assert_array_less(mu_n_min, mu_n)

    mu_p_max = np.ones(mu_p.shape) * 600. # cm^2/(V*s)
    mu_p_min = np.ones(mu_p.shape) * 400.

    testing.assert_array_less(mu_p, mu_p_max)
    testing.assert_array_less(mu_p_min, mu_p)

    # Diffusion coefficient
    (D_p, D_n) = physics.computeCarrierDiffusivities(mu_p, mu_n)

    # Sanity check
    testing.assert_array_less(D_p, D_n)

    D_n_max = np.ones(D_n.shape) * 39.6 # cm^2/s
    D_n_min = np.ones(D_n.shape) * 21.6

    testing.assert_array_less(D_n, D_n_max)
    testing.assert_array_less(D_n_min, D_n)

    D_p_max = np.ones(D_p.shape) * 16.2 # cm^2/s
    D_p_min = np.ones(D_n.shape) * 10.8 

    testing.assert_array_less(D_p, D_p_max)
    testing.assert_array_less(D_p_min, D_p)

    # Relative reciprocal diffusivity
    (y_p, y_n) = physics.computeRelativeReciprocalDiffusivities(D_p, D_n)

    y_max = np.ones(y_p.shape) * 2.0
    y_min = np.ones(y_n.shape) * 0.0

    testing.assert_array_less(y_min, y_p)
    testing.assert_array_less(y_p, y_max)

    testing.assert_array_less(y_min, y_n)
    testing.assert_array_less(y_n, y_max)

def test_integral_values(physics_fixture: Physics):
    physics = physics_fixture

    y = physics.mesh.nodes
    int_y_analytic = 1/2 * np.power(physics.mesh.nodes, 2)

    int_y_func = np.vectorize(lambda i: np.trapz(y[:i+1], physics.mesh.nodes[:i+1]))
    indices = np.linspace(0, physics.mesh.nodes.size - 1, physics.mesh.nodes.size, dtype=int)

    int_y_numerical = int_y_func(indices)

    testing.assert_allclose(int_y_analytic, int_y_numerical, atol=1e-9)

    int_y_func = np.vectorize(lambda i: np.trapz(y[i:], physics.mesh.nodes[i:]))

    int_y_numerical = int_y_func(indices)

    testing.assert_allclose(int_y_analytic[-1] - int_y_analytic, int_y_numerical, atol=1e-9)

def test_recombination(physics_fixture: Physics):
    physics = physics_fixture

    phi_p = 0; phi_n = 0; psi = 0
    u = physics.computeRecombinationRate(phi_p, phi_n, psi)

    assert u == 0

    psi = 10.0
    u = physics.computeRecombinationRate(phi_p, phi_n, psi)

    assert u == 0

    phi_p = 13.0
    psi = 13.0
    u = physics.computeRecombinationRate(phi_p, phi_n, psi)

    testing.assert_approx_equal(u, 84339, 0)

    u = np.zeros(physics.mesh.nodes.shape)
    J_r = physics.computeRecombinationCurrent(u)

    testing.assert_array_equal(J_r, np.zeros(J_r.size))

    u = physics.mesh.nodes
    J_r = physics.computeRecombinationCurrent(u)

    testing.assert_allclose(J_r[2], 4.64e-17 * physics.mesh.nodes[2]**2 / 2, atol=1e-24)

def test_quasi_fermi_potentials(physics_fixture: Physics):
    physics = physics_fixture

    psi = np.ones(physics.mesh.nodes.shape)
    gamma = np.ones(physics.mesh.nodes.shape)
    boundary_values = [0,0]
    phi_p = physics.computeHoleQuasiFermiPotential(psi, gamma, boundary_values)
    
    testing.assert_almost_equal(phi_p, np.zeros(physics.mesh.nodes.shape))

    boundary_values = [0,1]
    phi_p = physics.computeHoleQuasiFermiPotential(psi, gamma, boundary_values)

    testing.assert_almost_equal(phi_p[0], 0.)
    testing.assert_almost_equal(phi_p[-1], 1.)

    J_r = np.ones(physics.mesh.nodes.shape)
    phi_p = physics.computeHoleQuasiFermiPotential(psi, gamma, boundary_values, J_r)

    testing.assert_almost_equal(phi_p[0], 0.)
    testing.assert_almost_equal(phi_p[-1], 1.)

    boundary_values = [0,0]
    phi_n = physics.computeElectronQuasiFermiPotential(psi, gamma, boundary_values)

    testing.assert_almost_equal(phi_n, np.zeros(physics.mesh.nodes.shape))

    boundary_values = [0,1]
    phi_n = physics.computeElectronQuasiFermiPotential(psi, gamma, boundary_values)

    testing.assert_almost_equal(phi_n[0], 0.)
    testing.assert_almost_equal(phi_n[-1], 1.)

    phi_n = physics.computeElectronQuasiFermiPotential(psi, gamma, boundary_values, J_r)

    testing.assert_almost_equal(phi_n[0], 0.)
    testing.assert_almost_equal(phi_n[-1], 1.)