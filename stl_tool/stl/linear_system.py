import numpy as np

class ContinuousLinearSystem:
    """
    Very simple class to define a linear system storing the matrices A abd B
    """
    __counter = 0
    def __init__(self, A : np.ndarray, B:np.ndarray, C:np.ndarray = None , dt :float = 0.1, name : str = "LinearSystem"):
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

        if name == "LinearSystem":
            self.name = f"{name}_{ContinuousLinearSystem.__counter}"
            ContinuousLinearSystem.__counter += 1
        else:
            self.name = name
       
    
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
        From a given set of dimensions it returns the selction matrix (output matrix) that selects the elements at the given 
        dimensions. For example, for a vector of dimension 5, the list dims = [0,2] will return a matrix of size 5x2 that selects the first and third element of the vector.
        """
        if dims is None:
            return self.C
        else:
            try:
                return np.eye(self.A.shape[0])[dims,:]
            except IndexError:
                raise ValueError(f"Dimensions {dims} are out of range for the system with size {self.A.shape[0]}")
            
    def __str__(self):
            """
            String representation of the system
            """
            return f"Linear System with \nA:\n{self.A}\nB:\n{self.B}\nC:\n{self.C}\ndt: {self.dt}"


class MultiAgentSystem:
    def __init__(self, systems : list[ContinuousLinearSystem], dt : float = 0.1) -> None:
        """
        Create a multi-agent system from a list of linear systems.
        
        :param systems: list of linear systems
        :type systems: list[ContinuousLinearSystem]
        :param dt: sampling time used inside the controllers (like MPC,QP or PID)
        :type dt: float
        :return: multi-agent linear system
        :rtype: ContinuousLinearSystem
        """
        
        # self.A = np.block([[sys.A if i==j else np.zeros((sys.A.shape[0],systems[j].A.shape[1])) 
        #                 for j,sys in enumerate(systems)] 
        #                 for i,sys in enumerate(systems)])
        
        # self.B = np.block([[sys.B if i==j else np.zeros((sys.B.shape[0],systems[j].B.shape[1])) 
        #                 for j,sys in enumerate(systems)] 
        #                 for i,sys in enumerate(systems)])

        # self.C = np.block([[sys.C if i==j else np.zeros((sys.C.shape[0],systems[j].C.shape[1])) 
        #                 for j,sys in enumerate(systems)] 
        #                 for i,sys in enumerate(systems)])
        
        self.systems = systems
        self.dt      = dt
        self.system_dict = {sys.name: sys for sys in systems}
    
    def append_system(self, system : ContinuousLinearSystem):
        """
        Append a linear system to the multi-agent system.
        
        :param system: linear system to append
        :type system: ContinuousLinearSystem
        """
        self.systems.append(system)
        self.system_dict[system.name] = system

    def remove_system(self, system : ContinuousLinearSystem):
        """
        Remove a linear system from the multi-agent system.

        :param system: linear system to remove
        :type system: ContinuousLinearSystem
        """
        self.systems.remove(system)
        del self.system_dict[system.name]


    def relative_state_output_matrix(self, name_1: str, name_2: str, dims_1: list[int]|int = None, dims_2: list[int]|int = None) -> np.ndarray:
        """
        Returns the output matrix that computes the relative state between two systems in the multi-agent system.
        
        :param name_1: name of the first system
        :type name_1: str
        :param name_2: name of the second system
        :type name_2: str
        :param dims_1: dimensions of the first system to consider for the relative state, if None all dimensions are considered
        :type dims_1: list[int]|int, optional
        :param dims_2: dimensions of the second system to consider for the relative state, if None all dimensions are considered
        :type dims_2: list[int]|int, optional
        :return: output matrix that computes the relative state between the two systems. The output is of the form C_rel * x = x1 - x2, where x1 and x2 are the states of the two systems.
        :rtype: np.ndarray
        """
        
        sys_1 = self.system_dict.get(name_1)
        sys_2 = self.system_dict.get(name_2)
        
        if sys_1 is None or sys_2 is None:
            raise ValueError(f"Systems {name_1} and/or {name_2} not found in the multi-agent system")
        if len(dims_1) != len(dims_2):
            raise ValueError(f"Dimensions {dims_1} and {dims_2} must have the same length")
        
        C1 = sys_1.output_matrix_from_dimension(dims_1)
        C2 = sys_2.output_matrix_from_dimension(dims_2)
        
        n1 = sys_1.size_state
        n2 = sys_2.size_state
        
        idx1 = list(self.system_dict.keys()).index(name_1)
        idx2 = list(self.system_dict.keys()).index(name_2)
        
        total_state_size = sum(sys.size_state for sys in self.systems)
        
        C_rel = np.zeros((C1.shape[0], total_state_size))
        
        C_rel[:, idx1*n1:(idx1+1)*n1] = C1
        C_rel[:, idx2*n2:(idx2+1)*n2] = -C2
        
        return C_rel




class SingleIntegrator3d(ContinuousLinearSystem):
    """
    Simple 3D single integrator system
    """
    def __init__(self, dt = 0.1):
        """
        Initialize the single integrator system.
        """
        A = np.zeros((3,3))
        B = np.eye(3) 
        super().__init__(A, B, dt = dt)

class ISSDeputy(ContinuousLinearSystem):

    def __init__(self,dt = 0.1,r0 :float = 6771000):
        """
        Initialize the ISS deputy system with Clohessy-Wiltshire dynamics.
        """
        # Define the orbital radius of the chief satellite (ISS)
        A,B = self.cw_dynamics_matrix(r0)
        super().__init__(A, B, dt = dt)


    def cw_dynamics_matrix(self,r0, mu=3.986004418e14):
        """
        Returns the linear dynamics matrix A of the Clohessy-Wiltshire equations.
        
        Parameters:
            r0 : float
                Orbital radius of the chief satellite [meters]
            mu : float, optional
                Gravitational parameter of the Earth [m^3/s^2], default is Earth's
        
        Returns:
            A : ndarray
                6x6 dynamics matrix
        """
        n = np.sqrt(mu / r0**3)

        A = np.array([
            [0,    0,    0,    1,   0,   0],
            [0,    0,    0,    0,   1,   0],
            [0,    0,    0,    0,   0,   1],
            [3*n**2, 0,    0,    0,  2*n, 0],
            [0,    0,    0, -2*n,   0,   0],
            [0,    0, -n**2,  0,   0,   0]
        ])

        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        return A, B



def output_matrix(dims :list[int]|int, state_dim : int) :
    """
    From a given set of dimensions it returns the selction matrix (output matrix) that selects the elements at the given 
    dimensions. For example, for a vector of dimension 5, the list dims = [0,2] will return a matrix of size 5x2 that selects the first and third element of the vector.
    
    :param dims: list of dimensions to select
    :type dims: list[int]|int
    :param state_dim: dimension of the state space
    :type state_dim: int
    :return: output matrix
    :rtype: np.ndarray
    """
    
    if isinstance(dims, int):
        dims = [dims]
    
    if any(d < 0 or d >= state_dim for d in dims):
        raise ValueError(f"Dimensions {dims} are out of range for the system with size {state_dim}")
    
    return np.eye(state_dim)[dims,:]