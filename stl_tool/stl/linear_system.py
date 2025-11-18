import numpy as np
from scipy.linalg import expm

from .predicate_models import BoxPredicate, Polyhedron, Predicate
from dataclasses import dataclass
from ..polyhedron import box_polytope_matrices, box_polytope_matrices_from_bounds

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

        self.lbx : np.ndarray = None # state work space lower bounds
        self.ubx : np.ndarray = None # state work space upper bounds
        self.ubu : np.ndarray = np.ones(self.B.shape[1])*np.inf # input work space upper bounds
        self.lbu : np.ndarray = np.ones(self.B.shape[1])*np.inf # input work space lower bounds
       

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

    def add_state_naming(self, name: str,indices: list[int] | int):
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

    def get_dims_from_state_name(self, name: str) -> list[int]:
        """
        Get the indices of the state variables from their name

        :param name: name of the state variables
        :type name: str
        :return: indices of the state variables
        :rtype: list[int]
        """
        
        if name == "all":
            return list(range(self.A.shape[0]))
        elif name not in self.state_naming:
            raise ValueError(self._error_msg() + f"System {self.name} does not contain state {name}. You should should provide a naming for your state variables using the method 'add_state_naming'")

        return self.state_naming[name]
    
    def c2d(self) :
        """
        Discretize (A,B) using matrix exponential trick.
        """
        n = self.A.shape[0]
        m = self.B.shape[1]

        # Build block matrix
        M = np.zeros((n+m, n+m))
        M[:n, :n] = self.A
        M[:n, n:] = self.B

        # Matrix exponential
        Mexp = expm(M * self.dt)

        # Extract Ad, Bd
        Ad = Mexp[:n, :n]
        Bd = Mexp[:n, n:]
        return Ad, Bd

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
    
    @property
    def state_workspace(self) -> Polyhedron | None:
        """
        Return the state workspace as a polyhedron
        """
        if self.lbx is None or self.ubx is None:
            raise ValueError(self._error_msg() + "State workspace bounds are not set. Please use the method 'set_workspace_bounds' to set them.")

        A, b = box_polytope_matrices_from_bounds(self.lbx, self.ubx)

        return Polyhedron(A, b)

    @property
    def input_workspace(self) -> Polyhedron | None:
        """
        Return the input workspace as a polyhedron
        """
        if self.lbu is None or self.ubu is None:
            raise ValueError(self._error_msg() + "Input workspace bounds are not set. Please use the method 'set_workspace_bounds' to set them.")

        A, b = box_polytope_matrices_from_bounds(self.lbu, self.ubu)
        return Polyhedron(A, b)

    def sampling_time(self, dt):
        """
        Set the sampling time
        """
        self.dt = dt


    def output_matrix_from_dimension(self, dims:list[int] | int) -> np.ndarray:
        """
        From a given set of dimensions it returns the selction matrix (output matrix) that selects the elements at the given 
        dimensions. For example, for a vector of dimension 5, the list dims = [0,2] will return a matrix of size 5x2 that selects the first and third element of the vector.
        """

        if not isinstance(dims, list):
           dims = [dims]
        try:
            return np.eye(self.A.shape[0])[dims,:]
        except IndexError:
            raise ValueError(self._error_msg() +f"Dimensions {dims} are out of range for the system with size {self.A.shape[0]}")

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
            raise ValueError(self._error_msg()+ f"Name {name} not found in the state naming dictionary. You should should provide a naming for your state variables using the method 'add_state_naming'")
        
        dims = self.state_naming[name]
        return self.output_matrix_from_dimension(dims)
    


    def get_box_predicate_on_state_indices(self, size: np.ndarray, center: np.ndarray, dims: list[int] | int, predicate_name: str | None = None):
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
            raise ValueError(self._error_msg() + f"Size and center must have the same dimension as the number of selected state variables {C.shape[0]}")
        
        return BoxPredicate(size = size, center = center, dims = dims, state_dim = self.state_dim, systems_id = [self.id], name = predicate_name)
    

    def get_box_predicate_on_state_name(self, size: float | list[float], center: np.ndarray, state_name: str | None = None, predicate_name: str | None = None):
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
        
        if isinstance(size,float) :
            size = [size]*C.shape[0]
        if isinstance(size,int) :
            size = [float(size)]*C.shape[0]

        if len(center.flatten()) != C.shape[0] or len(size) != C.shape[0]:
            raise ValueError(f"Size and center must have the same dimension as the number of selected state variables {C.shape[0]}")
        
        try :
            dims_state = self.state_naming[state_name]
        except KeyError:
            raise ValueError(f"State name {state_name} not found in the state naming dictionary")

        return BoxPredicate(size = size, center = center, dims = dims_state, state_dim = self.state_dim, systems_id = [self.id], name = predicate_name)
    

    def set_workspace_bounds(self, ubx : list[float]| float, lbx : list[float]| float):
        """
        Set the workspace bounds for the system. This is useful for defining the state constraints in the MPC controller.
        
        :param ubx: upper bounds for the state variables
        :type ubx: list[float]| float
        :param lbx: lower bounds for the state variables
        :type lbx: list[float]| float
        """
        
        if isinstance(ubx, (int, float)):
            ubx = [float(ubx)]*self.state_dim
        if isinstance(lbx, (int, float)):
            lbx = [float(lbx)]*self.state_dim

        if len(ubx) != self.state_dim or len(lbx) != self.state_dim:
            raise ValueError(self._error_msg() + f"Upper and lower bounds must have the same dimension as the state variables {self.state_dim}")
        
        self.lbx = np.array(lbx).flatten()
        self.ubx = np.array(ubx).flatten()

    def set_input_bounds(self, ubu : list[float]| float, lbu : list[float]| float):
        """
        Set the input bounds for the system. This is useful for defining the input constraints in the MPC controller.
        
        :param ubu: upper bounds for the input variables
        :type ubu: list[float]| float
        :param lbu: lower bounds for the input variables
        :type lbu: list[float]| float
        """
        
        if isinstance(ubu, (int, float)):
            ubu = [float(ubu)]*self.input_dim
        if isinstance(lbu, (int, float)):
            lbu = [float(lbu)]*self.input_dim

        if len(ubu) != self.input_dim or len(lbu) != self.input_dim:
            raise ValueError(self._error_msg() + f"Upper and lower bounds must have the same dimension as the input variables {self.input_dim}")
        
        self.ubu = np.array(ubu).flatten()
        self.lbu = np.array(lbu).flatten()  




    def __str__(self):
            """
            String representation of the system
            """
            return f"Linear System with \nA:\n{self.A}\nB:\n{self.B}\nC:\n{self.C}\ndt: {self.dt}"

    def _error_msg(self):
        return f"System: {self.name} and id {self.id}: "
    
    def print_states_names(self):
        """
        Print the state naming dictionary
        """
        for name, indices in self.state_naming.items():
            print(f"{name}: {indices}")


class MultiAgentSystem:
    def __init__(self, systems : list[ContinuousLinearSystem], dt : float = 0.) -> None:
        """
        Create a multi-agent system from a list of linear systems.
        
        :param systems: list of linear systems
        :type systems: list[ContinuousLinearSystem]
        :param dt: sampling time used inside the controllers (like MPC,QP or PID)
        :type dt: float
        :return: multi-agent linear system
        :rtype: ContinuousLinearSystem
        """
        
        self.systems         : list[ContinuousLinearSystem] = systems
        self.dt              : float = dt if dt > 0. else min(sys.dt for sys in systems)
        self.systems_by_name : dict[str,ContinuousLinearSystem] = {sys.name: sys for sys in systems}

        # Define workspace
        try :
            workspace = self.systems[0].state_workspace
            for sys in self.systems[1:]:
                workspace = workspace * sys.state_workspace
        except Exception as e:
            raise ValueError(f"Something went wrong duing the computation of the joint workspace of the multi agent system."
            f" Please make sure that all the systems have their workspace defined using the method 'set_workspace_bounds'\n") from e

        # Define input space
        try :
            inputspace = self.systems[0].input_workspace
            for sys in self.systems[1:]:
                inputspace = inputspace * sys.input_workspace
        except Exception as e:
            raise ValueError(f"Something went wrong duing the computation of the joint input space of the multi agent system."
            f" Please make sure that all the systems have their input space defined using the method 'set_input_bounds'\n") from e

        self.workspace   = workspace
        self.inputbounds = inputspace

        # check if systems names are unique.
        for system in systems:
            for system2 in systems:
                if system != system2 and system.name == system2.name:
                    raise ValueError(f"System names must be unique. System {system.name} is repeated.")

    
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
    
    @property
    def A(self):
        """
        Return the state matrix
        """
        return np.block([[sys.A if i==j else np.zeros((sys.A.shape[0],self.systems[j].A.shape[1])) 
                        for j,sys in enumerate(self.systems)] 
                        for i,sys in enumerate(self.systems)])
    @property
    def B(self):
        """
        Return the input matrix
        """
        return np.block([[sys.B if i==j else np.zeros((sys.B.shape[0],self.systems[j].B.shape[1])) 
                        for j,sys in enumerate(self.systems)] 
                        for i,sys in enumerate(self.systems)])
    

    def c2d(self) :
        """
        Discretize (A,B) using matrix exponential trick.
        """
        n = self.A.shape[0]
        m = self.B.shape[1]

        # Build block matrix
        M = np.zeros((n+m, n+m))
        M[:n, :n] = self.A
        M[:n, n:] = self.B

        # Matrix exponential
        Mexp = expm(M * self.dt)

        # Extract Ad, Bd
        Ad = Mexp[:n, :n]
        Bd = Mexp[:n, n:]
        return Ad, Bd
    
    
    
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
    

        
    def get_system_by_name(self,name:str) -> ContinuousLinearSystem:
        """
        Get a system by its name.
        
        :return: system with the given name
        :rtype: ContinuousLinearSystem
        """
        try :
            return self.systems_by_name[name]
        except KeyError:
            raise ValueError(f"System with name {name} not found in the multi-agent system")
        

    def get_system_state_dims(self, system_name: str , state_name :str = "all") -> list[int]:
              
        system = self.systems_by_name.get(system_name)
        if system is None:
            raise ValueError(f"System {system_name} not found in the multi-agent system")

        dims = system.get_dims_from_state_name(state_name)

        idx = list(self.systems_by_name.keys()).index(system_name)
        from_dim = sum(sys.state_dim for sys in self.systems[:idx]) 
        dims = [from_dim + d for d in dims]

        # positions of the 
        return dims
    
    def get_system_input_dims(self, system_name: str):
        """
        Get the input dimensions of a system by its name.
        
        :param system_name: name of the system
        :type system_name: str
        :return: input dimensions of the system
        :rtype: list[int]
        """
        system = self.systems_by_name.get(system_name)
        if system is None:
            raise ValueError(f"System {system_name} not found in the multi-agent system")

        idx = list(self.systems_by_name.keys()).index(system_name)
        from_dim = sum(sys.input_dim for sys in self.systems[:idx]) 
        dims = [from_dim + d for d in range(system.input_dim)]

        return dims

    def _relative_state_output_matrix(self, system_1: str | int, system_2: str |int , dims_1: list[int]|int, dims_2: list[int]|int) -> np.ndarray:
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
        
        
        sys_1 = self.systems_by_name.get(system_1)
        sys_2 = self.systems_by_name.get(system_2)

        if sys_1 is None or sys_2 is None:
            raise ValueError(f"Systems {system_1} and/or {system_2} not found in the multi-agent system")
        
         
        if not isinstance(dims_1, list):
            dims_1 = [dims_1]
        if not isinstance(dims_2, list):
            dims_2 = [dims_2]

        if len(dims_1) != len(dims_2):
            raise ValueError(f"The number of dimensions for the two systems must be the same. Given {len(dims_1)} and {len(dims_2)}")
        
        C_1 = sys_1.output_matrix_from_dimension(dims_1)
        C_2 = sys_2.output_matrix_from_dimension(dims_2)
        
        n_1 = sys_1.state_dim
        n_2 = sys_2.state_dim

        idx_1 = list(self.systems_by_name.keys()).index(system_1)
        idx_2 = list(self.systems_by_name.keys()).index(system_2)

        from_dim_1 = sum(sys.state_dim for sys in self.systems[:idx_1])
        from_dim_2 = sum(sys.state_dim for sys in self.systems[:idx_2])

        total_state_size = sum(sys.state_dim for sys in self.systems)
        C_rel = np.zeros((C_1.shape[0], total_state_size))
        C_rel[:, from_dim_1:(from_dim_1+n_1)] = C_1
        C_rel[:, from_dim_2:(from_dim_2+n_2)] = -C_2

        return C_rel

    def _relative_state_output_matrix_by_name(self, system_1: str | int, system_2: str | int, state_name_1: str, state_name_2: str) -> np.ndarray:
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

        
        sys_1 = self.systems_by_name.get(system_1)
        sys_2 = self.systems_by_name.get(system_2)

        if sys_1 is None or sys_2 is None:
            raise ValueError(f"Systems {system_1} and/or {system_2} not found in the multi-agent system")
        
        dims_1 = sys_1.get_dims_from_state_name(state_name_1)
        dims_2 = sys_2.get_dims_from_state_name(state_name_2)

        C_rel = self._relative_state_output_matrix(system_1, system_2, dims_1, dims_2) 
        
        return C_rel



    def _state_output_matrix(self, system_name: str, dims: list[int]|int = None) -> np.ndarray:
        """
        Returns the output matrix that selects the state of a system in the multi-agent system.
        
        :param name: name of the system
        :type name: str | int
        :param dims: dimensions of the system to consider for the output, if None all dimensions are considered
        :type dims: list[int]|int, optional
        :return: output matrix that selects the state of the system. The output is of the form C * x = x_sys, where x_sys is the state of the system.
        :rtype: np.ndarray
        """
        sys = self.systems_by_name.get(system_name)

        if sys is None:
            raise ValueError(f"System {system_name} not found in the multi-agent system")
        
        if dims is None:
            dims = list(range(sys.state_dim))
        
        C_sys            = sys.output_matrix_from_dimension(dims)
        n_sys            = sys.state_dim
        idx_sys          = list(self.systems_by_name.keys()).index(system_name)

        total_state_size = sum(sys.state_dim for sys in self.systems)
        C                = np.zeros((C_sys.shape[0], total_state_size))
        from_index       = sum(sys.state_dim for sys in self.systems[:idx_sys])

        C[:, from_index:(from_index+n_sys)] = C_sys

        
        return C
    
    def _state_output_matrix_by_name(self, system_name: str, state_name: str) -> np.ndarray:
        """
        Returns the output matrix that selects the state of a system in the multi-agent system.
        
        :param name: name of the system
        :type name: str | int
        :param state_name: name of the state in the system
        :type state_name: str
        :return: output matrix that selects the state of the system. The output is of the form C * x = x_sys, where x_sys is the state of the system.
        :rtype: np.ndarray
        """
        sys = self.systems_by_name.get(system_name)

        if sys is None:
            raise ValueError(f"System {system_name} not found in the multi-agent system")
        
        dims = sys.get_dims_from_state_name(state_name)
        
        C = self._state_output_matrix(system_name, dims)
        
        return C
    

    def get_edge_box_predicate(self, system_1: str , system_2: str , dims_1: list[int]|int, dims_2: list[int]|int, desired_relative_state: np.ndarray, size: float | list[float]) -> BoxPredicate:
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

        sys_1 = self.systems_by_name.get(system_1)
        sys_2 = self.systems_by_name.get(system_2)
        desired_relative_state = desired_relative_state.flatten()
        
        if sys_1 is None or sys_2 is None:
            raise ValueError(f"Systems {system_1} and/or {system_2} not found in the multi-agent system")
        

        C_rel = self._relative_state_output_matrix(system_1, system_2, dims_1, dims_2)
        
        if len(desired_relative_state.flatten()) != C_rel.shape[0]:
            raise ValueError(f"Desired relative state must have the same dimension as the number of selected state variables {C_rel.shape[0]}. Given {len(desired_relative_state.flatten())}")
        

        ############################################################
        # Creation of the predicate 
        ###########################################################
        if not isinstance(size, list):
            size = float(size)
            size = np.array([size] *len(dims_1))
        else:
            size = np.array(size).flatten()
            if len(size) != len(desired_relative_state):
                raise ValueError(f"The size must be a {len(desired_relative_state)}D vector or a scalar, while the given predicate dimensions are ({len(size)})")


        A, b = box_polytope_matrices(desired_relative_state, size)

        polytope  = Polyhedron(A, b)
        predicate = Predicate(polytope=polytope, output_matrix=C_rel, systems_id=[sys_1.name, sys_2.name])

        return predicate
    

    def get_edge_box_predicate_by_name(self, system_1: str , system_2: str , state_name_1: str, state_name_2: str, desired_relative_state: np.ndarray, size: float | list[float]) -> BoxPredicate:
        """
        Create a box predicate on the relative state between two systems in the multi-agent system.
        
        :param system_name_1: name of the first system
        :type system_name_1: str | int
        :param system_name_2: name of the second system
        :type system_name_2: str | int
        :param state_name_1: name of the state in the first system
        :type state_name_1: str
        :param state_name_2: name of the state in the second system
        :type state_name_2: str
        :param desired_relative_state: desired relative state between the two systems
        :type desired_relative_state: np.ndarray
        :return: box predicate on the relative state between the two systems
        :rtype: BoxPredicate
        """

        sys_1 = self.systems_by_name.get(system_1)
        sys_2 = self.systems_by_name.get(system_2)
        
        if sys_1 is None or sys_2 is None:
            raise ValueError(f"Systems {system_1} and/or {system_2} not found in the multi-agent system")
        
        dims_1 = sys_1.get_dims_from_state_name(state_name_1)
        dims_2 = sys_2.get_dims_from_state_name(state_name_2)

        predicate = self.get_relative_formation_box_predicate(system_1, system_2, dims_1, dims_2, desired_relative_state, size)
        return predicate
    
    def get_relative_formation_box_predicate(self, *relative_fromations : "RelativeFormationTuple" ) :


        As = []
        bs = []

        for formation in relative_fromations:
            A, b = box_polytope_matrices(formation.center, formation.size)
            sys_1 = self.systems_by_name.get(formation.system_1)
            sys_2 = self.systems_by_name.get(formation.system_2)

            if sys_1 is None or sys_2 is None:
                raise ValueError(f"Systems {formation.system_1} and/or {formation.system_2} not found in the multi-agent system")

            if formation.dims_1 is None:
                dims_1 = sys_1.state_naming[formation.state_name_1]
            else:
                dims_1 = formation.dims_1
            if formation.dims_2 is None:
                dims_2 = sys_2.state_naming[formation.state_name_2]
            else:
                dims_2 = formation.dims_2

            C_rel = self._relative_state_output_matrix(formation.system_1, formation.system_2, dims_1, dims_2)
            As.append(A@C_rel)
            bs.append(b)

        systems = list(set([formation.system_1 for formation in relative_fromations] + [formation.system_2 for formation in relative_fromations])) # unique names of the systems


        A = np.vstack(As)
        b = np.hstack(bs)

        return Predicate(polytope=Polyhedron(A,b), systems_id=systems)
    
    def get_single_agent_box_predicate(self, system_name: str, dims: list[int]|int, center: np.ndarray, size: float | list[float]) -> BoxPredicate:
        """
        Create a box predicate on the state of a system in the multi-agent system.
        
        :param system_name: name of the system
        :type system_name: str | int
        :param dims: dimensions of the system to consider for the output
        :type dims: list[int]|int
        :param center: center of the box
        :type center: np.ndarray
        :param size: size of the box
        :type size: float | list[float]
        :return: box predicate on the state of the system
        :rtype: BoxPredicate
        """

        sys = self.systems_by_name.get(system_name)
        
        if sys is None:
            raise ValueError(f"System {system_name} not found in the multi-agent system")
        
        C_sys = self._state_output_matrix(system_name, dims)

        if len(center.flatten()) != C_sys.shape[0]:
            raise ValueError(f"Center must have the same dimension as the number of selected state variables {C_sys.shape[0]}")
        
        if not isinstance(size, list):
            size = float(size)
            size = np.array([size] *len(dims))
        else:
            size = np.array(size).flatten()
            if len(size) != len(center.flatten()):
                raise ValueError(f"The size must be a {len(center.flatten())}D vector or a scalar, while the given predicate dimensions are ({len(size)})")

        A,b = box_polytope_matrices(center.flatten(), size)
        polytope = Polyhedron(A@C_sys,b)


        return Predicate(polytope=polytope, systems_id=[sys.name])

    def get_single_agent_box_predicate_by_name(self, system_name: str, state_name: str, center: np.ndarray, size: float | list[float]) -> BoxPredicate:
        """
        Create a box predicate on the state of a system in the multi-agent system.
        
        :param system_name: name of the system
        :type system_name: str | int
        :param state_name: name of the state in the system
        :type state_name: str
        :param center: center of the box
        :type center: np.ndarray
        :param size: size of the box
        :type size: float | list[float]
        :return: box predicate on the state of the system
        :rtype: BoxPredicate
        """

        sys = self.systems_by_name.get(system_name)
        
        if sys is None:
            raise ValueError(f"System {system_name} not found in the multi-agent system")
        
        C_sys = self._state_output_matrix_by_name(system_name, state_name)

        if len(center.flatten()) != C_sys.shape[0]:
            raise ValueError(f"Center must have the same dimension as the number of selected state variables {C_sys.shape[0]}")
        
        if not isinstance(size, list):
            size = float(size)
            size = np.array([size] *len(sys.state_naming[state_name]))
        else:
            size = np.array(size).flatten()
            if len(size) != len(center.flatten()):
                raise ValueError(f"The size must be a {len(center.flatten())}D vector or a scalar, while the given predicate dimensions are ({len(size)})")

        A,b = box_polytope_matrices(center.flatten(), size)
        polytope = Polyhedron(A@C_sys,b)


        return Predicate(polytope=polytope, systems_id=[sys.name])
    
    def __iter__(self):
        # This makes the class iterable
        for system in self.systems:
            yield system


class SingleIntegrator3d(ContinuousLinearSystem):
    """
    Simple 3D single integrator system
    """
    def __init__(self, dt = 0.1, name: str = "SingleIntegrator3d"):
        """
        Initialize the single integrator system.
        """
        A = np.zeros((3,3))
        B = np.eye(3) 
        super().__init__(A, B, dt = dt, name = name + f"_{self.__counter}" if name == "SingleIntegrator3d" else name)
        self.add_state_naming('position', [0,1,2])
        self.add_state_naming('x', 0)
        self.add_state_naming('y', 1)
        self.add_state_naming('z', 2)

class SingleIntegrator2d(ContinuousLinearSystem):
    """
    Simple 2D single integrator system
    """
    def __init__(self, dt = 0.1, name: str = "SingleIntegrator2d"):
        """
        Initialize the single integrator system.
        """
        A = np.zeros((2,2))
        B = np.eye(2) 
        super().__init__(A, B, dt = dt, name = name + f"_{self.__counter}" if name == "SingleIntegrator2d" else name)
        self.add_state_naming('position', [0,1])
        self.add_state_naming('x', 0)
        self.add_state_naming('y', 1)

class DoubleIntegrator3d(ContinuousLinearSystem):
    """
    Simple 3D double integrator system
    """
    def __init__(self, dt = 0.1, name: str = "DoubleIntegrator3d"):
        """
        Initialize the double integrator system.
        """
        A = np.block([[np.zeros((3,3)), np.eye(3)],
                      [np.zeros((3,3)), np.zeros((3,3))]])
        B = np.block([[np.zeros((3,3))],
                      [np.eye(3)]])
        super().__init__(A, B, dt = dt, name = name + f"_{self.__counter}" if name == "DoubleIntegrator3d" else name)
        self.add_state_naming([0,1,2], 'position')
        self.add_state_naming([3,4,5], 'velocity')
        self.add_state_naming('x', 0)
        self.add_state_naming('y', 1)
        self.add_state_naming('z', 2)
        self.add_state_naming('vx', 3)
        self.add_state_naming('vy', 4)
        self.add_state_naming('vz', 5)

class DoubleIntegrator2d(ContinuousLinearSystem):
    """
    Simple 2D double integrator system
    """
    def __init__(self, dt = 0.1, name: str = "DoubleIntegrator2d"):
        """
        Initialize the double integrator system.
        """
        A = np.block([[np.zeros((2,2)), np.eye(2)],
                      [np.zeros((2,2)), np.zeros((2,2))]])
        B = np.block([[np.zeros((2,2))],
                      [np.eye(2)]])
        super().__init__(A, B, dt = dt, name = name + f"_{self.__counter}" if name == "DoubleIntegrator2d" else name)
        self.add_state_naming([0,1], 'position')
        self.add_state_naming([2,3], 'velocity')
        self.add_state_naming('x', 0)
        self.add_state_naming('y', 1)
        self.add_state_naming('vx', 2)
        self.add_state_naming('vy', 3)


class ISSDeputy(ContinuousLinearSystem):

    def __init__(self,dt = 0.1,r0 :float = 6771000, name: str = "ISSDeputy"):
        """
        Initialize the ISS deputy system with Clohessy-Wiltshire dynamics.
        """
        # Define the orbital radius of the chief satellite (ISS)
        A,B = self.cw_dynamics_matrix(r0)
        super().__init__(A, B, dt = dt, name = name + f"_{self.__counter}" if name == "ISSDeputy" else name)

        self.add_state_naming([0,1,2], 'position')
        self.add_state_naming([3,4,5], 'velocity')
        self.add_state_naming('x', 0)
        self.add_state_naming('y', 1)
        self.add_state_naming('z', 2)
        self.add_state_naming('vx', 3)
        self.add_state_naming('vy', 4)
        self.add_state_naming('vz', 5)


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

@dataclass
class RelativeFormationTuple:
    """
    Data class to store the parameters for a relative formation specification between two systems.
    """
    center       : np.ndarray
    size         : float | list[float]
    system_1     : str  
    system_2     : str  
    dims_1       : list[int] = None
    dims_2       : list[int] = None
    state_name_1 : str = None
    state_name_2 : str = None

    def __post_init__(self):
        self.center = self.center.flatten()
        
        if self.dims_1 is None and self.state_name_1 is None:
            raise ValueError("Either dims_1 or state_name_1 must be provided")
        
        if self.dims_2 is None and self.state_name_2 is None:
            raise ValueError("Either dims_2 or state_name_2 must be provided")
        
        if self.dims_1 is not None :
            if not isinstance(self.dims_1, list):
                self.dims_1 = [self.dims_1]
        
        if self.dims_2 is not None :
            if not isinstance(self.dims_2, list):
                self.dims_2 = [self.dims_2]
        
        if isinstance(self.size, (int, float)):
            self.size = [float(self.size)] * len(self.center)
        
        if self.dims_1 is not None and self.dims_2 is not None:
            if len(self.dims_1) != len(self.dims_2):
                raise ValueError(f"The number of dimensions for the two systems must be the same. Given {len(self.dims_1)} and {len(self.dims_2)}")
        
        if isinstance(self.size, list):
            if len(self.size) != len(self.center):
                raise ValueError(f"The size must be a {len(self.center)}D vector or a scalar, while the given predicate dimensions are ({len(self.size)})") 