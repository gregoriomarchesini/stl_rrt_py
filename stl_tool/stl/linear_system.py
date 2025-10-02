import numpy as np
from predicate_models import BoxPredicate, Polyhedron

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
        self.state_naming :dict[str, list[int]] = dict() # dictionary to store the naming of the state variables e.x. {'position': [0,1,2], 'velocity': [3,4,5]}

        if C is None:
            self.C = np.eye(A.shape[0])
        else:
            self.C = C

        # dimensionality checks
        assert self.A.shape[0] == self.A.shape[1], "Matrix A must be square"
        assert self.A.shape[0] == self.B.shape[0], "Matrix A and B must have compatible dimensions"
        assert self.A.shape[0] == self.C.shape[1], "Matrix A and C must have compatible dimensions"

        self.id = ContinuousLinearSystem.__counter

        if name == "LinearSystem":
            self.name = f"{name}_{self.id}"
        else:
            self.name = name

        # Increment the counter for the next instance
        ContinuousLinearSystem.__counter += 1
       
    
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

    def add_state_naming(self, indices: list[int] | int, name: str):
        """
        Add a name to state variables

        :param indices: indices of the state variables
        :type indices: list[int]
        :param name: name of the state variables
        :type name: str
        """

        if not isinstance(indices, list):
            indices = [indices]

        for index in indices:
            if index < 0 or index >= self.A.shape[0]:
                raise ValueError(f"Index {index} is out of range for the system with size {self.A.shape[0]}")
        
        self.state_naming[name] = indices

    @property
    def state_dim(self):
        """
        Return the size of the state
        """
        return self.A.shape[0]
    
    @property
    def input_dim(self):
        """
        Return the size of the input
        """
        return self.B.shape[1]
    
    def sampling_time(self, dt):
        """
        Set the sampling time
        """
        self.dt = dt
    
        
    def output_matrix_from_dimension(self,dims:list[int] |int ):
        """
        From a given set of dimensions it returns the selction matrix (output matrix) that selects the elements at the given 
        dimensions. For example, for a vector of dimension 5, the list dims = [0,2] will return a matrix of size 5x2 that selects the first and third element of the vector.
        """

        if not isinstance(dims, list):
           dims = [dims]
        try:
            return np.eye(self.A.shape[0])[dims,:]
        except IndexError:
            raise ValueError(f"Dimensions {dims} are out of range for the system with size {self.A.shape[0]}")

    def get_output_matrix_from_name(self, name: str):
        """
        From a given name it returns the selction matrix (output matrix) that selects the elements with the given name.
        For example, if the state variables are named as follows: { 'position': [0,1,2], 'velocity': [3,4,5] }, 
        then the name 'position' will return a matrix of size 3x6 that selects the first three elements of the vector.
        
        :param name: name of the state variables
        :type name: str
        :return: output matrix
        :rtype: np.ndarray
        """
        if name not in self.state_naming:
            raise ValueError(f"Name {name} not found in the state naming dictionary")
        
        dims = self.state_naming[name]
        return self.output_matrix_from_dimension(dims)


    def get_box_predicate_on_state_indices(self, size: np.ndarray, center: np.ndarray, dims: list[int] | int):
        """
        Create a box predicate on the state variables with the given indices.
        
        :param size: size of the box
        :type size: np.ndarray
        :param center: center of the box
        :type center: np.ndarray
        :param dims: indices of the state variables
        :type dims: list[int] | int
        :param name: name of the predicate
        :type name: str | None
        :return: box predicate
        :rtype: BoxPredicate
        """
        
        
        C = self.output_matrix_from_dimension(dims)
        if len(center.flatten()) != C.shape[0] or len(size) != C.shape[0]:
            raise ValueError(f"Size and center must have the same dimension as the number of selected state variables {C.shape[0]}")
        
        return BoxPredicate(size = size, center = center, dims = dims, state_dim = self.state_dim, systems_id = [self.id], name = None)
    

    def get_box_predicate_on_state_name(self, size: np.ndarray, center: np.ndarray, state_name: str | None = None):
        """
        Create a box predicate on the state variables with the given name.
        
        :param size: size of the box
        :type size: np.ndarray
        :param center: center of the box
        :type center: np.ndarray
        :param name: name of the state variables
        :type name: str
        :param predicate_name: name of the predicate
        :type predicate_name: str | None
        :return: box predicate
        :rtype: BoxPredicate
        """

        C = self.get_output_matrix_from_name(state_name)
        if len(center.flatten()) != C.shape[0] or len(size) != C.shape[0]:
            raise ValueError(f"Size and center must have the same dimension as the number of selected state variables {C.shape[0]}")
        
        dims_state = self.state_naming[state_name]

        return BoxPredicate(size = size, center = center, dims = dims_state, state_dim = self.state_dim, systems_id = [self.id], name = None)

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
        
        self.systems     : list[ContinuousLinearSystem] = systems
        self.dt          : float = dt
        self.system_dict : dict[int,ContinuousLinearSystem] = {sys.id: sys for sys in systems}

    
    @property
    def state_dim(self):
        """
        Return the size of the state
        """
        return sum(sys.state_dim for sys in self.systems)
    @property
    def input_dim(self):
        """
        Return the size of the input
        """
        return sum(sys.input_dim for sys in self.systems)
    

    def append_system(self, system : ContinuousLinearSystem):
        """
        Append a linear system to the multi-agent system.
        
        :param system: linear system to append
        :type system: ContinuousLinearSystem
        """
        self.systems.append(system)
        self.system_dict[system.id] = system

    def remove_system(self, system : ContinuousLinearSystem):
        """
        Remove a linear system from the multi-agent system.

        :param system: linear system to remove
        :type system: ContinuousLinearSystem
        """
        self.systems.remove(system)
        del self.system_dict[system.id]

    def relative_state_output_matrix(self, system_1: str | int, system_2: str | int, dims_1: list[int]|int, dims_2: list[int]|int) -> np.ndarray:
        """
        Returns the output matrix that computes the relative state between two systems in the multi-agent system.

        :param system_name_1: name of the first system
        :type system_name_1: str | int
        :param system_name_2: name of the second system
        :type system_name_2: str | int
        :param dims_1: dimensions of the first system to consider for the relative state, if None all dimensions are considered
        :type dims_1: list[int]|int, optional
        :param dims_2: dimensions of the second system to consider for the relative state, if None all dimensions are considered
        :type dims_2: list[int]|int, optional
        :return: output matrix that computes the relative state between the two systems. The output is of the form C_rel * x = x1 - x2, where x1 and x2 are the states of the two systems.
        :rtype: np.ndarray
        """
        
        if isinstance(system_1,str):
            for sys in self.systems:
                if sys.name == system_1:
                    system_1 = sys.id
                    break
            else:
                raise ValueError(f"System {system_1} not found in the multi-agent system")
        if isinstance(system_2,str):
            for sys in self.systems:
                if sys.name == system_2:
                    system_2 = sys.id
                    break
            else:
                raise ValueError(f"System {system_2} not found in the multi-agent system")
        
        sys_1 = self.system_dict.get(system_1)
        sys_2 = self.system_dict.get(system_2)
        if sys_1 is None or sys_2 is None:
            raise ValueError(f"Systems {system_1} and/or {system_2} not found in the multi-agent system")
        
        C_1 = sys_1.output_matrix_from_dimension(dims_1)
        C_2 = sys_2.output_matrix_from_dimension(dims_2)
        n_1 = sys_1.state_dim
        n_2 = sys_2.state_dim
        idx_1 = list(self.system_dict.keys()).index(system_1)
        idx_2 = list(self.system_dict.keys()).index(system_2)
        total_state_size = sum(sys.state_dim for sys in self.systems)
        C_rel = np.zeros((C_1.shape[0], total_state_size))
        C_rel[:, idx_1*n_1:(idx_1+1)*n_1] = C_1
        C_rel[:, idx_2*n_2:(idx_2+1)*n_2] = -C_2
        
        return C_rel

    def relative_state_output_matrix_by_name(self, system_1: str | int, system_2: str | int, state_name_1: str, state_name_2: str) -> np.ndarray:
        """
        Returns the output matrix that computes the relative state between two systems in the multi-agent system.

        :param system_name_1: name of the first system
        :type system_name_1: str | int
        :param system_name_2: name of the second system
        :type system_name_2: str | int
        :param state_name_1: name of the state in the first system
        :type state_name_1: str
        :param state_name_2: name of the state in the second system
        :type state_name_2: str
        :return: output matrix that computes the relative state between the two systems. The output is of the form C_rel * x = x1 - x2, where x1 and x2 are the states of the two systems.
        :rtype: np.ndarray
        """

        if isinstance(system_1,str):
            for sys in self.systems:
                if sys.name == system_1:
                    system_1 = sys.id
                    break
            else:
                raise ValueError(f"System {system_1} not found in the multi-agent system")
        if isinstance(system_2,str):
            for sys in self.systems:
                if sys.name == system_2:
                    system_2 = sys.id
                    break
            else:
                raise ValueError(f"System {system_2} not found in the multi-agent system")
        
        sys_1 = self.system_dict.get(system_1)
        sys_2 = self.system_dict.get(system_2)
        if sys_1 is None or sys_2 is None:
            raise ValueError(f"Systems {system_1} and/or {system_2} not found in the multi-agent system")
        
        C_1 = sys_1.get_output_matrix_from_name(state_name_1)
        C_2 = sys_2.get_output_matrix_from_name(state_name_2)
        n_1 = sys_1.state_dim
        n_2 = sys_2.state_dim
        idx_1 = list(self.system_dict.keys()).index(system_1)
        idx_2 = list(self.system_dict.keys()).index(system_2)
        total_state_size = sum(sys.state_dim for sys in self.systems)
        C_rel = np.zeros((C_1.shape[0], total_state_size))
        C_rel[:, idx_1*n_1:(idx_1+1)*n_1] = C_1
        C_rel[:, idx_2*n_2:(idx_2+1)*n_2] = -C_2
        
        return C_rel

        
    
    
    def state_output_matrix(self, system_name: str | int, dims: list[int]|int = None) -> np.ndarray:
        """
        Returns the output matrix that selects the state of a system in the multi-agent system.
        
        :param name: name of the system
        :type name: str | int
        :param dims: dimensions of the system to consider for the output, if None all dimensions are considered
        :type dims: list[int]|int, optional
        :return: output matrix that selects the state of the system. The output is of the form C * x = x_sys, where x_sys is the state of the system.
        :rtype: np.ndarray
        """
        if isinstance(system_name,str):
            for sys in self.systems:
                if sys.name == system_name:
                    system_name = sys.id
                    break
            else:
                raise ValueError(f"System {system_name} not found in the multi-agent system")
        
        sys = self.system_dict.get(system_name)
        if sys is None:
            raise ValueError(f"System {system_name} not found in the multi-agent system")
        
        if dims is None:
            dims = list(range(sys.state_dim))
        
        C_sys            = sys.output_matrix_from_dimension(dims)
        n_sys            = sys.state_dim
        idx_sys          = list(self.system_dict.keys()).index(system_name)
        total_state_size = sum(sys.state_dim for sys in self.systems)
        C                = np.zeros((C_sys.shape[0], total_state_size))
        C[:, idx_sys*n_sys:(idx_sys+1)*n_sys] = C_sys
        
        return C
    

    def get_relative_formation_between(self, system_1: str | int, system_2: str | int, dims_1: list[int]|int, dims_2: list[int]|int, desired_relative_state: np.ndarray, size: float | list[float]) -> BoxPredicate:
        """
        Create a box predicate on the relative state between two systems in the multi-agent system.
        
        :param system_name_1: name of the first system
        :type system_name_1: str | int
        :param system_name_2: name of the second system
        :type system_name_2: str | int
        :param dims_1: dimensions of the first system to consider for the relative state
        :type dims_1: list[int]|int
        :param dims_2: dimensions of the second system to consider for the relative state
        :type dims_2: list[int]|int
        :param desired_relative_state: desired relative state between the two systems
        :type desired_relative_state: np.ndarray
        :return: box predicate on the relative state between the two systems
        :rtype: BoxPredicate
        """
        
        C_rel = self.relative_state_output_matrix(system_1, system_2, dims_1, dims_2)
        if len(desired_relative_state.flatten()) != C_rel.shape[0]:
            raise ValueError(f"Desired relative state must have the same dimension as the number of selected state variables {C_rel.shape[0]}")

        return BoxPredicate(size = size, center = desired_relative_state, dims = None, state_dim = self.state_dim, systems_id = [system_1, system_2], name = None)



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
        self.add_state_naming([0,1,2], 'position')
        self.add_state_naming([0], 'x')
        self.add_state_naming([1], 'y')
        self.add_state_naming([2], 'z')


class ISSDeputy(ContinuousLinearSystem):

    def __init__(self,dt = 0.1,r0 :float = 6771000):
        """
        Initialize the ISS deputy system with Clohessy-Wiltshire dynamics.
        """
        # Define the orbital radius of the chief satellite (ISS)
        A,B = self.cw_dynamics_matrix(r0)
        super().__init__(A, B, dt = dt)

        self.add_state_naming([0,1,2], 'position')
        self.add_state_naming([3,4,5], 'velocity')
        self.add_state_naming([0], 'x')
        self.add_state_naming([1], 'y')
        self.add_state_naming([2], 'z')
        self.add_state_naming([3], 'vx')
        self.add_state_naming([4], 'vy')
        self.add_state_naming([5], 'vz')


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