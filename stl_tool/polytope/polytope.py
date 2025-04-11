import numpy as np
import cdd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

"""This file was originally developed by Mikael Johansson and is part of the openMPC library"""

class Polytope:
    def __init__(self, A, b):
        # Normalize the polytope inequalities to the form a^Tx <= 1




        # # centering the polytope 
        # if not np.all(b >= 0.) :
        #     self.center = - np.pinv(A) @ b
        # else:
        #     self.center = np.zeros(A.shape[1])

        A, b = self.normalize(A, b)
        
        # Ensure that b is a 1D array
        b = b.flatten()

        self.A = A 
        self.b = b 

        # Convert A and b into CDDlib format (H-representation)
        mat          = np.hstack([b.reshape(-1, 1), -A])
        self.cdd_mat = cdd.matrix_from_array(mat, rep_type=cdd.RepType.INEQUALITY)
        self.poly    = cdd.polyhedron_from_matrix(self.cdd_mat)
        
        
        self._is_open  :bool         = None
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
        if self._is_open is None: # after finding the v representaiton you can know if it open
            self._get_V_representation()
        
        return self._is_open

    @property
    def vertices(self)-> np.ndarray:
        if self._vertices is None:
            self._get_V_representation()
        return self._vertices
    
    @property
    def rays(self) -> np.ndarray:
        if self._rays is None:
            self._get_V_representation()
        return self._rays

    def normalize(self, A, b):
        """
        Normalize the inequalities to the form a^T x <= 1.

        Args:
            A (numpy.ndarray): Inequality matrix of shape (m, n).
            b (numpy.ndarray): Right-hand side vector of shape (m,) or (m, 1).

        Returns:
            (numpy.ndarray, numpy.ndarray): Normalized A and b.
        """
        # Ensure b is a 1D array
        b = b.flatten()
        
        # Check if dimensions are compatible
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has {A.shape[0]} rows, but b has {b.shape[0]} elements.")
        
        
        # Avoid division by zero
        b_norm       = np.where(b == 0, 1, b)  # Use 1 to avoid division by zero
        A_normalized = A/np.abs(b_norm[:, np.newaxis])
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
        Get the vertices (V-representation) of the polytope.
        """
        generators_lib = cdd.copy_generators(self.poly)
        generators     = np.array(generators_lib.array)
        linearities    = np.array(list(generators_lib.lin_set)) # It tells which rays are allowed to be also negative (https://people.inf.ethz.ch/fukudak/cdd_home/cddlibman2021.pdf) pp. 4
        
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
    
    
    def plot(self, ax=None, color='b', edgecolor='k', alpha=1.0, linestyle='-', showVertices=False):
        """
        Plot the polytope using Matplotlib.

        Args:
            ax: Matplotlib axis object.
            color: Fill color of the polytope.
            edgecolor: Color of the polytope edges (if different from color).
            alpha: Transparency of the fill.
            linestyle: Line style for the edges.
            showVertices: Boolean, whether to show the vertices as points.
        """
        # Get vertices for plotting
        if not self.is_open:
            vertices = self.vertices
        else:
            big_number = 10000
            vertices   = np.vstack(( self.vertices, self.vertices + big_number*self.rays))

        if len(vertices) == 0:
            return  # Nothing to plot if no vertices

        vertices = np.array(vertices)
        dim = vertices.shape[1]

        if dim > 3:
            raise ValueError("Cannot plot polytopes with more than 3 dimensions.")

         # If no axis is provided, create a new figure
        if ax is None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection='3d')  # Ensure 3D projection
            else:
                ax = fig.add_subplot(111)
        else:
            # If the provided axis is 2D but the polytope is 3D, convert it to 3D
            if dim == 3 and not hasattr(ax, "get_proj"):
                fig = ax.figure  # Get the current figure
                # save limits
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                fig.delaxes(ax)  # Remove the old 2D axis
                ax = fig.add_subplot(111, projection='3d')  # Replace with a 3D axis

                # set limits as per teh previois axis 
                ax.set_xlim(xlim)   
                ax.set_ylim(ylim)
                ax.set_zlim(xlim)
        try:
            hull = ConvexHull(vertices)
            hull_vertices = vertices[hull.vertices]
        except:
            hull_vertices = vertices  # Fallback to raw vertices if hull fails

        if dim == 2:
            # 2D Polytope Plotting
            ax.plot(np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
                    np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
                    color=edgecolor, linestyle=linestyle)

            ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=color, alpha=alpha)

            if showVertices:
                ax.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'ro')

        elif dim == 3:
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
        Args:
            dims (list): Indices of the dimensions to keep.
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
        projected_polytope = Polytope(reduced_A[:, dims], reduced_b)

        # Remove redundancies from the new polytope
        projected_polytope.remove_redundancies()

        return projected_polytope

    def intersect(self, other):
        """
        Compute the intersection of two polytopes (self and other).
        Returns:
            A new Polytope representing the intersection.
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
        Returns:
            A new Polytope representing the intersection.
        """
        return self.intersect(other)

    def contains(self, x, tol=1e-8):
        """
        Check if a point x is inside the polytope.
        Args:
            x: A point as a numpy array.
        Returns:
            True if the point is inside the polytope, False otherwise.
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
        Args:
            x: A point as a numpy array.
        Returns:
            True if the point is inside the polytope, False otherwise.
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
        Get creates a polytope with the given dimension out of the lower dimension polytope by expanding the A matrix 
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
    

class BoxNd(Polytope):
    """
    Box in nD space
    """

    def __init__(self,n_dim:int,  size:float|list[float], center:np.ndarray = None) -> None:
        """
        Args:
            center : np.ndarray
                center of the box
            size : float
                size of the box

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
        Args:
            x : float
                x coordinate of the box
            y : float
                y coordinate of the box
            w : float
                width of the box
            h : float
                height of the box

        """
        self.x = x
        self.y = y
        super().__init__(n_dim = 2, size = size, center = np.array([x,y]))


class Box3d(BoxNd):

    def __init__(self, x:float, y:float, z:float, size: float|list[float]) -> None:
        """
        Args:
            x : float
                x coordinate of the box
            y : float
                y coordinate of the box
            z : float
                z coordinate of the box
        """
        self.x = x
        self.y = y
        self.z = z
        center = np.array([x, y, z])
        super().__init__(n_dim = 3, size = size, center = center)


class Geq2d(Polytope):
    def __init__(self, coordinate : str, bound : float) :
        """
        Args:
            coordinate : str
                coordinate of the box
            bound : float
                bound of the box
        """
        if coordinate == "x":
            A = np.array([[-1., 0]])
        elif coordinate == "y":
            A = np.array([[0., -1]])
        else:
            raise ValueError("coordinate should be either x or y")
        
        b = -np.array([bound])

        super().__init__(A, b)

class Leq2d(Polytope):
    def __init__(self, coordinate : str, bound : float) :
        """
        Args:
            coordinate : str
                coordinate of the box
            bound : float
                bound of the box
        """
        if coordinate == "x":
            A = np.array([[1., 0]])
        elif coordinate == "y":
            A = np.array([[0., 1]])
        else:
            raise ValueError("coordinate should be either x or y")
        
        b = np.array([bound])

        super().__init__(A, b)


def concatenate_diagonally(*polytopes: Polytope ):
    """
    Concatenate polytopes diagonally to create a new polytope.
    Args:
        *polytopes: List of Polytope objects to concatenate.
    Returns:
        A new Polytope object representing the concatenated polytope.
    """
    A = np.diag([polytope.A for polytope in polytopes])
    b = np.concatenate([polytope.b for polytope in polytopes])
    # Ensure b is a 1D array
    b = b.flatten()
    

    return Polytope(A, b)




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