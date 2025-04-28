import numpy as np
import cdd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

### This file was originally developed by Mikael Johansson and is part of the openMPC library
### It has been modified by the stl_tool team to fit the needs of the library

class Polytope:
    """ 
    General Polytope class. Every polytope is represented as Conv(V) + Cone(R) where V are the vertices and R are the rays.
    The polytope is defined by the H-representation Ax-b <= 0. The vertices and rays are computed using cddlib.
    """
    def __init__(self, A : np.ndarray , b:np.ndarray) -> None:
        """
        :param A: Matrix A of the polyhedron Ax-b <= 0
        :type A: np.ndarray
        :param b: Vector b of the polyhedron Ax-b <= 0
        :type b: np.ndarray
        """
        
       
        b = b.flatten()
        self.A = A 
        self.b = b 

        # Convert A and b into CDDlib format (H-representation)
        mat          = np.hstack([b.reshape(-1, 1), -A])
        self.cdd_mat = cdd.matrix_from_array(mat, rep_type=cdd.RepType.INEQUALITY)
        self.poly    = cdd.polyhedron_from_matrix(self.cdd_mat)
        
        
        self._is_open  :bool         = None  # defines if the given polytope is open (thus a polyhedra)
        self._vertices :np.ndarray   = None  # vertices are enumarated along the rows
        self._rays     :np.ndarray   = None  # rays are enumarated along the rows

        

    @property
    def num_hyperplanes(self):
        """
        Number hyperplanes (constraints) of the polytope.
        """
        return self.A.shape[0]
    
    @property
    def num_dimensions(self):
        """
        Number of dimensions of the polytope.
        """
        return self.A.shape[1]
    
    @property
    def is_open(self) -> bool:
        """
        Check if the polytope is open (i.e., it has rays).
        """
        if self._is_open is None: # after finding the v representaiton you can know if it is open
            self._get_V_representation()
        
        return self._is_open

    @property
    def vertices(self)-> np.ndarray:
        """
        Get the vertices of the polytope defining the CONV(V) part of the polytope.
        """
        if self._vertices is None:
            self._get_V_representation()
        return self._vertices
    
    @property
    def rays(self) -> np.ndarray:
        """
        Get the rays of the polytope defining the CONE(R) part of the polytope.
        """
        if self._rays is None:
            self._get_V_representation()
        return self._rays

    def normalize(self, A : np.ndarray, b:np.ndarray):
        """
        Normalize the inequalities to the form a^T x <= 1.

       :param A: Coefficient matrix.
       :type A: numpy.ndarray
       :param b: Right-hand side vector.
       :type b: numpy.ndarray
       :return: Normalized A and b.
       :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """

        # Ensure b is a 1D array
        b = b.flatten()
        
        # Check if dimensions are compatible
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has {A.shape[0]} rows, but b has {b.shape[0]} elements.")
        
        
        # Avoid division by zero
        b_norm       = np.where(b == 0, 1, b)  # Use 1 to avoid division by zero
        A_normalized = A/np.abs(b_norm)[:,np.newaxis]
        b_normalized = b/np.abs(b_norm)

        return A_normalized, b_normalized

    # Other methods (plot, projection, intersect, etc.) remain unchanged


    def __repr__(self):
        """
        Custom print method to display the H-representation of the polytope in a user-friendly format,
        ensuring that the string length (including minus sign) is the same for all entries of A, 
        and that entries of b are also equally spaced.
        """
        # Determine the maximum width for each element in A (for consistent alignment)
        A_max_len = max(len(f"{coef:.5f}") for row in self.A for coef in row)
        b_max_len = max(len(f"{bval:.5f}") for bval in self.b)

        repr_str = "Hyperplane representation of polytope\n"
        
        for i in range(self.A.shape[0]):
            if i == 0:
                repr_str += "  [["  # Start the first line with '[['
            else:
                repr_str += "   ["  # Subsequent lines with proper indentation
            
            # Format each row of A with the determined max width
            row_str = "  ".join(f"{coef:{A_max_len}.5f}" for coef in self.A[i])
            
            # Add the corresponding b value with consistent width
            if i == self.A.shape[0] - 1:
                repr_str += f"{row_str}] x <= [{self.b[i]:{b_max_len}.5f}]\n"  # Last line
            else:
                repr_str += f"{row_str}] |    [{self.b[i]:{b_max_len}.5f}]\n"
        
        return repr_str


    def _get_V_representation(self):
        """
        Computes vertices and rays of the polytope. If rays are present then the polytope is declared as open (a polyhedron)
        """
        generators_lib = cdd.copy_generators(self.poly)
        generators     = np.array(generators_lib.array)
        linearities    = np.array(list(generators_lib.lin_set)) # It tells which rays are allowed to be also negative (https://people.inf.ethz.ch/fukudak/cdd_home/cddlibman2021.pdf) pp. 4
        
        if not len(generators) :
            raise ValueError("The polytope is empty. No vertices or rays found.")
        vertices_indices = generators[:,0] == 1. # vertices are only the generators with 1 in the first column (https://people.inf.ethz.ch/fukudak/cdd_home/cddlibman2021.pdf) pp. 4
        
        vertices      = generators[vertices_indices, 1:]   # Skip the first column (homogenizing coordinate) # vertices are returns a 2d array where vertices are enumareted aloing the rows
        rays          = generators[~vertices_indices, 1:]  # Skip the first column (homogenizing coordinate) # rays are returns a 2d array where rays are enumareted along the rows
        
        if len(linearities) :
            inverted_rays = -generators[linearities,1:]        # these are the rays that should be also considered in the oppositive direction
            rays          = np.vstack((rays, inverted_rays))   # Add the inverted rays to the rays list

        # Determine if the polytope is open or closed
        if len(rays) > 0:
            self._is_open = True # not all generators are vertices (some of them are rays indeed)
        else :
            self._is_open = False
        
        # cache vertices and rays to save computation
        self._vertices = vertices
        self._rays     = rays
        

    def volume(self):
        """
        Compute the volume of the polytope. Handles 1D intervals separately.
        """
        
        vertices = self.vertices

        # If there are no vertices, the volume is zero
        if len(vertices) == 0:
            return 0.0

        # Handle 1D intervals (polytope in 1D)
        if vertices.shape[1] == 1:
            return np.max(vertices) - np.min(vertices)

        # Handle higher-dimensional polytopes (2D and above)
        try:
            hull = ConvexHull(vertices)
            return hull.volume
        except Exception as e:
            print(f"Error computing volume: {e}")
            return 0.0
    
    
    def plot(self, ax=None, color='b', edgecolor='k', alpha=1.0, linestyle='-', showVertices=False, projection_dims : list[int] = []):
        """
        Plot the polytope using Matplotlib. Only 2d or 3d plots are allowed. 

        :param ax: Matplotlib axis to plot on. If None, a new figure is created.
        :type ax: matplotlib.axes.Axes
        :param color: Fill color for the polytope.
        :type color: str
        :param edgecolor: Edge color for the polytope.
        :type edgecolor: str
        :param alpha: Transparency level for the fill color.
        :type alpha: float
        :param linestyle: Line style for the edges.
        :type linestyle: str
        :param showVertices: If True, show the vertices of the polytope.
        :type showVertices: bool
        :param projection_dims: List of dimensions to project onto. If empty, the first 2 or 3 dimensions are used.
        :type projection_dims: list[int]
        """

        # Get vertices for plotting.
        if not self.is_open:
            vertices = self.vertices
        else: # we draw an hyperplane by shoting the vertices in the direction of the rays
            big_number = 100
            new_vertices = []
            for vertex in self.vertices:
                shiften_vertices = vertex + big_number * self.rays
                new_vertices.append(shiften_vertices)
            # create new list of vertices with the shifted ones
            vertices = np.vstack((self.vertices, *new_vertices))

        if len(vertices) == 0:
            return  # Nothing to plot if no vertices.

        # define plotting dimensions.
        n_dim = vertices.shape[1]

        if len(projection_dims) == 0:
            projection_dims = [i for i in range(min(3, n_dim))] # plot at most 3d.
        else:
            projection_dims = list(set(projection_dims)) # remove duplicates.

        # check that dimensions required are feasible for projection.
        if len(projection_dims) > 3:
            raise ValueError("Cannot plot polytopes with more than 3 dimensions.")
        if len(projection_dims) > n_dim:
            raise ValueError(f"Polytope does not have enough dimensions for the given list.")
        
        try :
            C = selection_matrix_from_dims(n_dims = n_dim, selected_dims=projection_dims) # this matrix 
        except IndexError as e:
            raise ValueError(f"you provide at least one dimension outside the dimensionality of the polytope. Raise exception issue: {e}")

        vertices = (C @ vertices.T).T  # shape [num_vertices, dim]
        
        # If no axis is provided, create a new figure
        if ax is None:
            fig = plt.figure()
            if len(projection_dims) == 3:
                ax = fig.add_subplot(111, projection='3d')  # Ensure 3D projection
            else:
                ax = fig.add_subplot(111)
        else :
            if len(projection_dims) == 3 and not hasattr(ax, "set_zlabel"):
                raise ValueError("The provided axis is not a 3D axis. Please provide a 3D axis for 3D plotting. Consider provide the argument projected_dims=[0,1] to plot on 2D. Otherwise provde 3d axis.")
        
        # create convex hull
        try:
            hull          = ConvexHull(vertices)
            hull_vertices = vertices[hull.vertices]
        except Exception as e:
            hull_vertices = vertices  # Fallback to raw vertices if hull fails
            print(f"Error computing convex hull: {e}")


        if len(projection_dims)== 2:
            # 2D Polytope Plotting
            ax.plot(np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
                    np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
                    color=edgecolor, linestyle=linestyle)

            ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=color, alpha=alpha)

            if showVertices:
                ax.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'ro')

        elif len(projection_dims) == 3:
            # 3D Polyhedron Plotting
            faces = [hull.simplices[i] for i in range(len(hull.simplices))]
            poly3d = [[vertices[i] for i in face] for face in faces]
            ax.add_collection(Poly3DCollection(poly3d, facecolors=color, linewidths=1, edgecolors=edgecolor, alpha=alpha))

            if hasattr(ax, "set_zlabel"):
                ax.set_zlabel("Z")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        return ax

    def remove_redundancies(self):
        """
        Simplify the polytope by removing redundant constraints using matrix_canonicalize.
        """
        # Make a copy of the matrix
        mat_copy = cdd.matrix_copy(self.cdd_mat)
        
        # Canonicalize the matrix to remove redundant constraints
        cdd.matrix_canonicalize(mat_copy)

        # Update A and b after redundancy removal
        reduced_array = np.array(mat_copy.array)
        self.A = -reduced_array[:, 1:]
        self.b = reduced_array[:, 0]
        
        # Update the cdd matrix as well
        self.cdd_mat = mat_copy

    def projection(self, dims):
        """
        Project the polytope onto the subspace of 'dims' using cddlib block_elimination.

        :param dims: List of dimensions to project onto.
        :type dims: list[int]
        :return: A new Polytope object representing the projected polytope.
        :rtype: Polytope
        """

        A = self.A
        b = self.b

        # Convert A and b into the matrix required for CDD
        mat = np.hstack([b.reshape(-1, 1), -A])
        cdd_mat = cdd.matrix_from_array(mat.tolist(), rep_type=cdd.RepType.INEQUALITY)

        # We need to shift the dims because column 0 is b in [b -A]
        num_vars = A.shape[1]
        cols_to_eliminate = set(range(1, num_vars + 1)) - set(d + 1 for d in dims)

        # Perform block elimination using cddlib
        reduced_mat = cdd.block_elimination(cdd_mat, cols_to_eliminate)

        # Extract the reduced A and b from the resulting matrix
        reduced_array = np.array(reduced_mat.array)  # Access the matrix using .array
        reduced_A = -reduced_array[:, 1:]  # Take the matrix part (excluding the first column)
        reduced_b = reduced_array[:, 0]    # First column corresponds to b

        # Create a new polytope in the projected space
        try:
            projected_polytope = Polytope(reduced_A[:, dims], reduced_b)
        except IndexError as e:
            raise ValueError(f"Invalid dimensions for projection of polytope that is {self.num_dimensions}-dimensional: {dims}. Error: {e}")
        # Remove redundancies from the new polytope
        projected_polytope.remove_redundancies()

        return projected_polytope

    def intersect(self, other):
        """
        Compute the intersection of two polytopes (self and other).

        :param other: Another Polytope object to intersect with.
        :type other: Polytope
        :return: A new Polytope object representing the intersection.
        :rtype: Polytope
        """
        # Combine the inequalities from both polytopes
        combined_A = np.vstack([self.A, other.A])
        combined_b = np.concatenate([self.b, other.b])

        # Create a new polytope representing the intersection
        intersected_polytope = Polytope(combined_A, combined_b)

        # Remove redundancies from the intersection result
        intersected_polytope.remove_redundancies()

        return intersected_polytope
    
    def __and__(self, other):
        """
        Compute the intersection of two polytopes using the '&' operator.
        
        :param other: Another Polytope object to intersect with.
        :type other: Polytope
        :return: A new Polytope object representing the intersection.
        :rtype: Polytope
        """
        return self.intersect(other)

    def contains(self, x, tol=1e-8):
        """
        Check if a point x is inside the polytope.
        
        :param x: A point as a numpy array.
        :type x: numpy.ndarray
        :param tol: Tolerance for numerical stability.
        :type tol: float
        :return: True if the point is inside the polytope, False otherwise.
        :rtype: bool
        """
        # Check if Ax <= b holds for the point x
        return np.all(np.dot(self.A, x) <= self.b + tol)

    def sample_random(self, num_samples=1):
        """
        Returns array of random samples in shape [dim, num_samples].
        For each sample:
        - Selects 3 random vertices
        - Takes a convex combination using Dirichlet distribution
        - Adds a random non-negative linear combination of rays


        :param num_samples: Number of random samples to generate.
        :type num_samples: int
        :return: Array of random samples in shape [dim, num_samples].
        :rtype: numpy.ndarray
        """

        vertices = self.vertices.T  # shape [dim, num_vertices]
        rays = self.rays.T          # shape [dim, num_rays]
        dim = vertices.shape[0]

        samples = np.zeros((dim, num_samples))

        for i in range(num_samples):
            # Choose 3 distinct random vertices
            idx = np.random.choice(vertices.shape[1], size=3, replace=False)
            chosen_vertices = vertices[:, idx]  # shape [dim, 3]

            # Dirichlet distribution gives a convex combination (sum = 1)
            weights = np.random.dirichlet(alpha=np.ones(3))  # shape [3]

            # Convex combination
            point = chosen_vertices @ weights  # shape [dim]

            # Add a random combination of rays
            if rays.shape[1] > 0:
                ray_weights = np.random.rand(rays.shape[1]) * 1e5
                point += rays @ ray_weights  # shift point along rays

            samples[:, i] = point

        return samples

    def __contains__(self, x):
        """
        Check if a point x is inside the polytope.
        """
        return self.contains(x)


    def __eq__(self, other, tol=1e-7):
        """
        Check if two polytopes define the same set by comparing their canonical forms.
        This handles ordering and small numerical differences.
        """
        # Canonicalize both polytopes
        self_canon = cdd.matrix_copy(self.cdd_mat)
        other_canon = cdd.matrix_copy(other.cdd_mat)
        
        cdd.matrix_canonicalize(self_canon)
        cdd.matrix_canonicalize(other_canon)

        # Convert the canonicalized matrices to arrays for comparison
        self_array = np.array(self_canon.array)
        other_array = np.array(other_canon.array)

        # Sort the rows of both arrays (for consistent comparison)
        self_sorted = np.array(sorted(self_array, key=lambda row: tuple(row)))
        other_sorted = np.array(sorted(other_array, key=lambda row: tuple(row)))

        # Compare the sorted arrays with a tolerance
        return (self_sorted.shape == other_sorted.shape) and np.allclose(self_sorted, other_sorted, atol=tol)
    
    def get_inflated_polytope_to_dimension(self, dim:int):
        """
        For a given polytope Ax-b, it infaltez the polytope by padding the matrix A with zero such that 
        new_A = [A|0] for a given number of zeros.

        :param dim: Desired dimension of the inflated polytope.
        :type dim: int
        :return: A new Polytope object representing the inflated polytope.
        :rtype: Polytope
        """

        # Get the current dimensions of the polytope
        current_dim = self.num_dimensions

        # If the current dimension is already equal to the desired dimension, return the original polytope
        if current_dim == dim:
            return self
        elif current_dim > dim:
            raise ValueError("The current dimension is greater than the desired dimension. Cannot inflate. use projection instead")
        else :
            # Create a new A matrix with the desired number of dimensions
            new_A = np.zeros((self.A.shape[0], dim))

            # Fill in the existing A matrix values
            new_A[:, :current_dim] = self.A

        return Polytope(new_A, self.b)
    

    def cross(self,other : "Polytope") -> "Polytope":
   
        """
        Cartesian product of two polytopes. Polytopes are stacked vertically such that 

        .. math: 
            A = diag(A1, A2, ... An)

            b = [b1, b2, ... bn]
        

        :param polytopes: List of Polytope objects to be combined.
        :type polytopes: list[Polytope]
        :return: A new Polytope object representing the Cartesian product.
        :rtype: Polytope
    
        """
        A1 = self.A
        A2 = other.A

        block_matrix = np.block([
                                    [A1, np.zeros((A1.shape[0], A2.shape[1]))],  # Top-left block: A1
                                    [np.zeros((A2.shape[0], A1.shape[1])), A2]   # Bottom-right block: A2
                                ])
        
        b = np.hstack((self.b, other.b))
        # Ensure b is a 1D array
        b = b.flatten()
        
        return Polytope(block_matrix , b)
    
    def __mul__(self, other:"Polytope") -> "Polytope" :
        """
        Cartesian product of two polytopes using the '+' operator.
        
        :param other: Another Polytope object to combine with.
        :type other: Polytope
        :return: A new Polytope object representing the Cartesian product.
        :rtype: Polytope
        """
        return self.cross(other)
    
    def __rmul__(self, other:"Polytope") -> "Polytope" :
        """
        Cartesian product of two polytopes using the '*' operator.
        
        :param other: Another Polytope object to combine with.
        :type
        :rtype: Polytope
        """
        return self.cross(other)
    
    
    def __pow__(self, exponent):
        """
        Raise the polytope to a power using the '^' operator. This is equivalent to taking the Cartesian product of the polytope with itself 'exponent' times.
        
        :param exponent: The exponent to raise the polytope to.
        :type exponent: int
        :return: A new Polytope object representing the Cartesian product.
        :rtype: Polytope
        """
        if not isinstance(exponent, int) or exponent < 1:
            raise ValueError("Exponent must be a positive integer.")
        
        result = self
        for _ in range(exponent - 1):
            result = result.cross(self)
        
        return result
    
    def __matmul__(self, C : np.ndarray) -> "Polytope":
        """
        Matrix multiplication of the polytope with a vector or matrix (usually a selection matrix).
        
        :param other: A vector or matrix to multiply with.
        :type other: numpy.ndarray
        :return: The result of the multiplication.
        :rtype: numpy.ndarray
        """

        A = self.A @ C

        return Polytope(A, self.b)
    

class BoxNd(Polytope):
    """
    Box in nD space
    """

    def __init__(self,n_dim:int,  size:float|list[float], center:np.ndarray = None) -> None:
        """
        Initialize a box in nD space. The size and center length must match the number of dimensions. The size can be a simple float, in this case the box will have save width in all dimensions.
        
        :param n_dim: Number of dimensions of the box
        :type n_dim: int
        :param size: Size of the box. If a single value is provided, it is used for all dimensions.
        :type size: float or list[float]
        :param center: Center of the box. If None, it is set to the origin.
        :type center: np.ndarray
        """

        if center is None:
            center = np.zeros(n_dim)
            
        self.center = center
        self.n_dim  = n_dim
        
        if isinstance(size, (int, float)):
            self.size = np.array([size] * n_dim)
        else:
            self.size = np.array(size)

        # check dimensions 
        if len(self.size) != n_dim:
            raise ValueError(f"size should be of length {n_dim}. Given : {len(self.size)}")
        # check center dimensions
        if len(center) != n_dim:
            raise ValueError(f"center should be of length {n_dim}. Given : {len(center)}")
        

        # create the polytope
        A = np.vstack((np.eye(n_dim), -np.eye(n_dim)))
        b = np.hstack((self.size,self.size)) / 2 + A @ self.center

        super().__init__(A, b)


class Box2d(BoxNd):

    def __init__(self, x:float, y:float, size:float|list[float] ) -> None:
        """
        :params x: x coordinate of the box
        :type x: float
        :params y: y coordinate of the box
        :type y: float
        :params size: size of the box [width,height]. If a single value is provided, it is used for all dimensions.
        :type size: float or list[float]
        """
        self.x = x
        self.y = y
        super().__init__(n_dim = 2, size = size, center = np.array([x,y]))


class Box3d(BoxNd):

    def __init__(self, x:float, y:float, z:float, size: float|list[float]) -> None:
        """
        :params x: x coordinate of the box
        :type x: float
        :params y: y coordinate of the box
        :type y: float
        :params z: z coordinate of the box
        :type z: float
        :params size: size of the box [width,height,depth]. If a single value is provided, it is used for all dimensions.
        :type size: float or list[float]
        """
        self.x = x
        self.y = y
        self.z = z
        center = np.array([x, y, z])
        super().__init__(n_dim = 3, size = size, center = center)


class Icosahedron(Polytope):


    def __init__(self, radius=1.0, x=0.0, y=0.0, z=0.0):
        H,b,_,_ = self._icosahedron_h_representation(radius, center = np.array([x,y,z]))
        super().__init__(H, b)
        


    def _compute_plane_equation(self,v1, v2, v3):
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

    def _icosahedron_h_representation(self,radius=1.0, center=np.zeros(3), ellipsoid_axes=None):
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
            normal, b = self._compute_plane_equation(v1, v2, v3)
            H_list.append(normal)
            b_list.append(b)

        # Convert to numpy arrays
        H = np.array(H_list)
        b = np.array(b_list)

        return H, b, verts, faces




def cartesian_product(*polytopes: Polytope ):
    """
    Cartesian product of two polytopes. Polytopes are stacked vertically such that 

    .. math: 
        A = diag(A1, A2, ... An)

        b = [b1, b2, ... bn]
    

    :param polytopes: List of Polytope objects to be combined.
    :type polytopes: list[Polytope]
    :return: A new Polytope object representing the Cartesian product.
    :rtype: Polytope
   
    """
    A = np.diag([polytope.A for polytope in polytopes])
    b = np.concatenate([polytope.b for polytope in polytopes])
    # Ensure b is a 1D array
    b = b.flatten()
    

    return Polytope(A, b)

def selection_matrix_from_dims(n_dims :int , selected_dims : list[int]|int ) :
    """
    Creates a matrix to project a given vector of dimension n_dim to the selected dimensions, such that :math:'y = Cx' where x is the vector of dimension n_dim and y is the vector of dimension len(selected_dims).

    :param n_dims: Number of dimensions of the original vector.
    :type n_dims: int
    :param selected_dims: List of dimensions to select. If a single integer is provided, it is converted to a list.
    :type selected_dims: list[int] or int
    :return: Selection matrix of shape (len(selected_dims), n_dims).
    :rtype: numpy.ndarray
    """
    if isinstance(selected_dims, int):
        selected_dims = [selected_dims]
    
    for dim in selected_dims:
        if dim >= n_dims or dim < 0:
            raise IndexError(f"Dimension {dim} is not a valid selection for a state with {n_dims} dimension(s)")

    selection_matrix = np.zeros((len(selected_dims), n_dims))
    for i, dim in enumerate(selected_dims):
        selection_matrix[i, dim] = 1

    return selection_matrix


if __name__ == "__main__":
    
    
    # A = np.array([[-1., 0],[0.,-1]])
    # b = np.array([1.,1.])

    # polytope = Polytope(A, b)
    # vertices = polytope.vertices

    # print(vertices)
    # print(polytope.is_open)
    # print(polytope.rays)
    

    fig,ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # print("Vertices of the polytope:")
    # print(vertices)
    # polytope.plot()

    box = Box2d(0, 0, 1, 1)
    box.plot(ax=ax, color='red', alpha=0.5)
    samples = box.sample_random(2000)
    # scatter
    ax.scatter(samples[0], samples[1], color='blue', alpha=0.5)

    plt.show()