import numpy as np

from .logic import Predicate
from ..polytope import Polytope



class Geq(Predicate):
    """
    Predicate defining a lower bound on the state as x[dims] >= b. The list dims represents the dimension over which the predicate is defined.
    """
    def __init__(self, dims : list[int] | int, bound:float, name:str | None = None) :
        """
        :param dims: list of dimensions to apply the predicate
        :type dims: list[int]
        :param bound: lower bound
        :type bound: float
        :param name: name of the predicate
        :type name: str
        """

        if isinstance(dims, int):
            dims = [dims]
        

        A = np.eye(len(dims))
        b = np.array([bound]*len(dims))
        
        # polytope is always defined as Ax<= b
        polytope = Polytope(-A,-b)
        super().__init__(polytope = polytope, dims = dims, name = name)
        

class Leq(Predicate):
    """
    Predicate defining an upper bound on the state as x[dims] <= b. The list dims represents the dimension over which the predicate is defined.
    """
    def __init__(self, dims : list[int] | int, bound:float, name:str | None = None) :
        """
        :param dims: list of dimensions to apply the predicate
        :type dims: list[int]
        :param bound: upper bound
        :type bound: float
        :param name: name of the predicate
        :type name: str
        """

        if isinstance(dims, int):
            dims = [dims]
        

        A = np.eye(len(dims))
        b = np.array([bound]*len(dims))
        
        # polytope is always defined as Ax<= b
        polytope = Polytope(A,b)
        super().__init__(polytope = polytope, dims = dims, name = name)



class BoxBound(Predicate):
    """
    Represents a box region in the given dimensions of the state space. The list dims represents the dimension over which the predicate is defined.
    """

    def __init__(self, dims: list[int], size: float|list[float], center: np.ndarray = None, name:str | None = None):
        """
        :param dims: list of dimensions to apply the predicate
        :type dims: list[int]
        :param size: size of the box
        :type size: float or list[float]
        :param center: center of the box
        :type center: np.ndarray
        :param name: name of the predicate
        :type name: str
        """
        
        if center is None:
            center = np.zeros(len(dims))
        else:
            center = np.array(center).flatten()
            if len(center) != len(dims):
                raise ValueError(f"The center must be a {len(dims)}D vector when the given predicate dimensions are ({len(dims)}). Given center: {center}")
        
        if not isinstance(size, list):
            size = float(size)
            size = np.array([size] *len(dims))
        else:
            size = np.array(size).flatten()
            if len(size) != len(dims):
                raise ValueError(f"The size must be a {len(dims)}D vector when the given predicate dimensions are ({len(dims)}). Given size: {size}")
       

        b     = np.hstack((size,size)) / 2
        A     = np.vstack((np.eye(len(dims)), -np.eye(len(dims))))
        b_vec = b + A @ center  # Ensuring proper half-space representation

        polytope = Polytope(A, b_vec)
        super().__init__(polytope = polytope, dims = dims, name = name)

class BoxBound2d(BoxBound):
    """
    Polytope representing a 2D box on the first two dimension of the state space. 
    This is just a convenience class for BoxBound. It is equivalent to BoxBound(dims=[0, 1], size=size, center=center)
    """

    def __init__(self, size: float|list[float], center: np.ndarray = None, name:str | None = None):
        """
        :param size: size of the box
        :type size: float or list[float]
        :param center: center of the box
        :type center: np.ndarray
        :param name: name of the predicate
        :type name: str
        """
        super().__init__(dims=[0, 1], size=size, center=center, name=name)

class BoxBound3d(BoxBound):
    """
    Polytope representing a 3D box on the first three dimension of the state space.
    This is just a convenience class for BoxBound. It is equivalent to BoxBound(dims=[0, 1, 2], size=size, center=center)
    """
    def __init__(self, size: float|list[float], center: np.ndarray = None, name:str | None = None):
        """
        :param size: size of the box
        :type size: float or list[float]
        :param center: center of the box
        :type center: np.ndarray
        :param name: name of the predicate
        :type name: str
        """
        super().__init__(dims=[0, 1, 2], size=size, center=center, name=name)


class IcosahedronPredicate(Predicate):


    def __init__(self, radius=1.0, x=0.0, y=0.0, z=0.0 , name:str | None = None, dims: list[int] = [0, 1, 2]):
        """ The standard application of the Icosahedron is on three dimensions """

        if len(dims) != 3:
            raise ValueError(f"Icosahedron predicate is only defined in 3D. Given dimensions: {dims}. Soon we will try to expand to more dimensions")
        
        H,b,_,_ = self._icosahedron_h_representation(radius, center = np.array([x,y,z]))
        pp = Polytope(H, b)
        super().__init__(polytope = pp, dims = dims, name = name)
        

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








if __name__ == "__main__":
    pass