import pytest
import numpy as np
import numpy.testing as testing

from mesh import Mesh
from physics import Physics
from constants import Material

@pytest.fixture
def physics_fixture():
    mesh = Mesh(6, 10.)
    return Physics(mesh)

def test_shape_function(physics_fixture: Physics):
    physics = physics_fixture
    mesh = physics.mesh

    derivatives = physics.computeShapeFunctionDerivateTerms()

    testing.assert_almost_equal(derivatives, 1 / mesh.dl * np.array([[1, -1],[-1, 1]]))

    values = physics.computeShapeFunctionValueTerms()

    testing.assert_almost_equal(values, mesh.dl * np.array([[1, 0],[0, 1]]))

def test_boundary_values(physics_fixture: Physics):
    physics = physics_fixture
    mesh = physics.mesh

    doping_profile = physics.computeDopingProfile()

    testing.assert_equal(doping_profile[0], -Material.N_A)
    testing.assert_equal(doping_profile[-1], Material.N_D)

    boundary_values = physics.computeBoundaryValues(doping_profile, 0)