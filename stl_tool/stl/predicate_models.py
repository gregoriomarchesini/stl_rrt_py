import numpy as np

from .logic import Predicate
from ..polytope import Polytope



class GEQ(Predicate):
    """
    Greater than predicate in the form x[index] >= b
    """
    def __init__(self, dim :int, index:int, bound:float, name:str | None = None) :

        a        = np.zeros(dim)
        a[index] = 1.
        b        = bound
        
        # polytope is always defined as Ax<= b
        polytope = Polytope(-a,-b)
        super().__init__(polytope,name)
        

class LEQ2d(Predicate):
    """
    Less than predicate in the form x[index] <= b
    """
    def __init__(self, dim :int, index:int, bound:float, name:str | None = None) :

        a        = np.zeros(dim)
        a[index] = 1.
        b        = bound

        polytope = Polytope(a,b)
        super().__init__(polytope,name)



class BoxPredicate(Predicate):
    """
    Polytope representing an n-dimensional box.
    """

    def __init__(self, n_dim: int, size: float, center: np.ndarray = None, name:str | None = None):
        
        if center is None:
            center = np.zeros(n_dim)
            
        center = center.flatten()  # Ensure center is a 1D array
        if center.shape[0] != n_dim:
            raise ValueError(f"The center must be a {n_dim}D vector")

        b = size / 2
        A = np.vstack((np.eye(n_dim), -np.eye(n_dim)))
        b_vec = b + A @ center  # Ensuring proper half-space representation


        polytope = Polytope(A, b_vec)
        super().__init__(polytope,name)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Example usage
    dim = 3
    index = 1
    bound = 5.0

    geq_pred = GEQ(dim, index, bound)
    leq_pred = LEQ2d(dim, index, bound)
    
    
    geq_pred.plot()

    # box_pred = BoxPredicate(n_dim=3, size=2.0, center=np.array([1.0, 1.0, 1.0]))
    # print("Box Predicate:", box_pred.polytope)

    plt.show()