import numpy as np

class ContinuousLinearSystem:
    """
    Very simple class to define a linear system storing the matrices A abd B
    """
    def __init__(self, A : np.ndarray, B:np.ndarray, C:np.ndarray = None , dt :float = 0.1):
        """
        :param A: state matrix
        :type A: np.ndarray
        :param B: input matrix
        :type B: np.ndarray
        :param C: output matrix
        :type C: np.ndarray
        :param dt: sampling time used inside the controllers (like MPC,QP or PID)
        :type dt: float
        """
        
        self.A  = A # 
        self.B  = B
        self.dt = dt # samling time for the later discretization of a linear system

        if C is None:
            self.C = np.eye(A.shape[0])
        else:
            self.C = C

        # dimensionality checks
        assert self.A.shape[0] == self.A.shape[1], "Matrix A must be square"
        assert self.A.shape[0] == self.B.shape[0], "Matrix A and B must have compatible dimensions"
        assert self.A.shape[0] == self.C.shape[1], "Matrix A and C must have compatible dimensions"
       
    
    def controllability_matrix(self) -> np.ndarray:
        """
        Compute the controllability matrix
        
        :return: controllability matrix
        :rtype: np.ndarray
        """
        
        n = self.A.shape[0]
        m = self.B.shape[1]
        ctrb_matrix = np.zeros((n, n * m))
        for i in range(n):
            AiB = np.linalg.matrix_power(self.A, i) @ self.B  # shape (n, m)
            ctrb_matrix[:, i * m : (i + 1) * m] = AiB
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
    
        
    def output_matrix_from_dimension(self,dims:list[int] |int = None):
        """
        Return the output matrix
        """
        if dims is None:
            return self.C
        else:
            try:
                return np.eye(self.A.shape[0])[dims,:]
            except IndexError:
                raise ValueError(f"Dimensions {dims} are out of range for the system with size {self.A.shape[0]}")