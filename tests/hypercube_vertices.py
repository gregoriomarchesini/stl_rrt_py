
import numpy as np


def fast_hypercube_vertices(n_dim, center=None, size=1.0):
    """
    More efficient method using binary encoding.
    """
    num_vertices = 2 ** n_dim
    # Binary representation: bits of integers
    bits = np.arange(num_vertices, dtype=np.uint32)[:, None]
    signs = 2 * ((bits >> np.arange(n_dim)) & 1) - 1  # map 0→-1, 1→1

    if center is None:
        center = np.zeros(n_dim)
    return center + (size / 2) * signs

n = 20
v = fast_hypercube_vertices(n_dim=20)
print(v.shape)