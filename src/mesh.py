import numpy as np
import constants

class Mesh:

    def __init__(self, num_nodes, length):
        length = length / constants.Material.L_D

        self.num_nodes = num_nodes
        self.num_edges = num_nodes - 1
        self.length = length
        self.mid_point = length / 2.
        self.dl = length / float(self.num_edges)
        self.nodes = np.linspace(0, length, num_nodes)
        self.edges = np.linspace(0, self.num_edges, num_nodes, dtype=int).repeat(2, axis=0)[1:-1].reshape((-1,2))
        self.x_avg = np.mean(self.nodes[self.edges], axis=1)