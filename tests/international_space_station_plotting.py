import numpy as np
from scipy.spatial import ConvexHull
from stl_tool.polytope import Polytope

def compute_plane_equation(v1, v2, v3):
    """
    Compute the plane equation (H, b) from the triangle defined by vertices v1, v2, v3.
    H will be a normal vector, and b will be the distance to the origin.
    """
    # Vectors along the plane
    v1_v2 = v2 - v1
    v1_v3 = v3 - v1

    # Normal vector to the plane (cross product)
    normal = np.cross(v1_v2, v1_v3)

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Distance from the origin to the plane (dot product)
    b = np.dot(normal, v1)

    return normal, b

def icosahedron_h_representation(radius=1.0, center=np.zeros(3), ellipsoid_axes=None):
    """
    Generate the icosahedron in H-representation (Hx <= b).
    If ellipsoid_axes are provided, scale the icosahedron into an ellipsoid.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Vertices of an icosahedron centered at origin
    verts = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ])

    # Normalize to unit sphere and scale
    verts = verts / np.linalg.norm(verts[0]) * radius
    
    # If ellipsoid_axes is provided, scale the vertices
    if ellipsoid_axes:
        verts[:, 0] *= ellipsoid_axes[0]  # Scale x-axis
        verts[:, 1] *= ellipsoid_axes[1]  # Scale y-axis
        verts[:, 2] *= ellipsoid_axes[2]  # Scale z-axis

    # Translate to the desired center
    verts += center

    # Faces (indices into verts array)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])

    # Calculate the half-space representation (H, b) for each face
    H_list = []
    b_list = []
    for face in faces:
        v1, v2, v3 = verts[face]
        normal, b = compute_plane_equation(v1, v2, v3)
        H_list.append(normal)
        b_list.append(b)

    # Convert to numpy arrays
    H = np.array(H_list)
    b = np.array(b_list)

    return H, b, verts, faces



# Example: Generate the H-representation for a sphere (icosahedron)
radius = 1.0  # unit radius
center = np.array([0, 0, 0])  # center at origin

# Generate ellipsoid by stretching along different axes
ellipsoid_axes = [1.5, 1.0, 2.0]  # Different scaling for x, y, z axes

H, b, verts, faces = icosahedron_h_representation(radius=radius, center=center, ellipsoid_axes=ellipsoid_axes)

print("H matrix:")
print(H)
print("b vector:")
print(b)

pp = Polytope(H, b)


pp.plot(alpha=0.1)
# # Plot the icosahedron and its polyhedron (H-representation)
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the icosahedron
# collection = Poly3DCollection(verts[faces], alpha=0.3, edgecolor='k')
# ax.add_collection3d(collection)

# # # Plot the bounding ellipsoid
# # plot_polyhedron(ax, verts, faces)

# ax.set_box_aspect([1, 1, 1])
# plt.tight_layout()
plt.show()
