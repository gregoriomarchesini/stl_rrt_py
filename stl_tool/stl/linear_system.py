import numpy as np

class ContinuousLinearSystem:
    """
    Very simple class to define a linear system storing the matrices A abd B
    """
    def __init__(self, A, B,dt :float = 0.1):
        self.A = A # 
        self.B = B
        self.dt = dt # samling time for the later discretization of a linear system

    
    def controllability_matrix(self):
        """
        Compute the controllability matrix
        """
        n = self.A.shape[0]
        m = self.B.shape[1]
        ctrb_matrix = np.zeros((n, n * m))
        for i in range(n):
            for j in range(m):
                ctrb_matrix[:, i * m + j] = np.linalg.matrix_power(self.A, i) @ self.B[:, j]
        return ctrb_matrix
    
    def is_controllable(self):
        """
        Check if the system is controllable
        """
        # Check if the system is controllable
        n = self.A.shape[0]
        rank = np.linalg.matrix_rank(self.controllability_matrix())
        return rank == n
   
    @property
    def size_state(self):
        """
        Return the size of the state
        """
        return self.A.shape[0]
    
    @property
    def size_input(self):
        """
        Return the size of the input
        """
        return self.B.shape[1]
    
    def sampling_time(self, dt):
        """
        Set the sampling time
        """
        self.dt = dt
    
