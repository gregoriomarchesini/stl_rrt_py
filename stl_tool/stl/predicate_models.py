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
            if center.shape[0] != len(dims):
                raise ValueError(f"The center must be a {len(dims)}D vector when the given predicate dimensions are ({len(dims)})")
        
        if isinstance(size, float):
            size = np.array([size] * 2*len(dims))
        else:
            size = np.hstack((np.array(size).flatten(),np.array(size).flatten()))
            if size.shape[0] != len(dims):
                raise ValueError(f"The size must be a {len(dims)}D vector when the given predicate dimensions are ({len(dims)})")
       

        b     = size / 2
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


if __name__ == "__main__":
    pass