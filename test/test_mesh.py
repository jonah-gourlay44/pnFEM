import pytest
from mesh import Mesh

class MeshProps:
    MESH_LENGTH = 0.1 # cm
    NUM_NODES = 11

@pytest.fixture
def mesh_fixture():
    return Mesh(MeshProps.NUM_NODES, MeshProps.MESH_LENGTH)

def test_mesh_attributes(mesh_fixture):
    
    assert len(mesh_fixture.nodes) == MeshProps.NUM_NODES
    assert len(mesh_fixture.edges) == MeshProps.NUM_NODES - 1