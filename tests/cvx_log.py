import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Given matrix A and vector b
A = np.array([
    [ 1.33333333, -0.        ,  0.18064516],
    [-0.        ,  1.33333333,  0.18064516],
    [-1.33333333, -0.        ,  0.18064516],
    [-0.        , -1.33333333,  0.18064516]
])
b = np.array([2.66666667, 2.66666667, 2.66666667, 2.66666667])

# Convert Ax <= b into halfspaces form: -A x + b >= 0 → [-A | b]
halfspaces = np.hstack((-A, b.reshape(-1, 1)))

# # Find a strictly interior point using linear programming
# row_norms = np.linalg.norm(A, axis=1)
# A_ext = np.hstack([A, row_norms.reshape(-1, 1)])
# c = np.array([0, 0, 0, -1])  # minimize -t to maximize t
# res = linprog(c, A_ub=A_ext, b_ub=b, bounds=[(None, None)]*3 + [(0, None)])

# if not res.success:
#     raise RuntimeError("Failed to find a strictly interior point.")

# interior_point = res.x[:3]

# Compute the halfspace intersection
hs = HalfspaceIntersection(halfspaces, np.array([0.,0.,0.]))

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Convex hull to define faces
hull = ConvexHull(hs.intersections)
for simplex in hull.simplices:
    triangle = [hs.intersections[v] for v in simplex]
    poly = Poly3DCollection([triangle], alpha=0.5, facecolor='cyan', edgecolor='k')
    ax.add_collection3d(poly)

# Plot the vertices
ax.scatter(hs.intersections[:, 0], hs.intersections[:, 1], hs.intersections[:, 2], color='k')

# Set axes limits
pts = hs.intersections
ax.set_xlim(pts[:,0].min() - 1, pts[:,0].max() + 1)
ax.set_ylim(pts[:,1].min() - 1, pts[:,1].max() + 1)
ax.set_zlim(pts[:,2].min() - 1, pts[:,2].max() + 1)

# Labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Polyhedron defined by Ax ≤ b')

plt.tight_layout()
plt.show()
