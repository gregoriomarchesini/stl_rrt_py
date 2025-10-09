import cvxpy as cp
import numpy as np
from   stl_tool.stl.linear_system import ContinuousLinearSystem,MultiAgentSystem



class Constraint:
    """
    Simple class to represent polyhedral constraints of the form Hx <= b.
    """
    def __init__(self, H : np.ndarray , b : np.ndarray, is_hard : bool =True, penalty_weight : float | None = None):

        """
        Constructor for the Constraint class.

        Define constraints of the form Hx <= b. Penality is given for soft constraints in MPC optimization.
        
        :param H: The matrix H in the constraint Hx <= b.
        :type H: np.ndarray
        :param b: The vector b in the constraint Hx <= b.
        :type b: np.ndarray
        :param is_hard: A boolean indicating whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for the constraint. Only used if is_hard is False.
        :type penalty_weight: float

        """
        
        
        self.H       = H
        self.b       = b.flatten()
        self.is_hard = is_hard
        self.penalty_weight = penalty_weight  # Only used if is_hard is False
        
        if self.penalty_weight is not None:
            if self.penalty_weight < 0:
                raise ValueError("penalty_weight must be non-negative.")

        if len(b) != H.shape[0]:
            raise ValueError("Number of rows in A must match the length of b. A has shape {} and b has length {}.".format(H.shape, len(b)))

    def to_polytope(self):
        """Returns the polytope representation."""
        return self.H, self.b




class TimedConstraint(Constraint):
    """
    Class to represent timed constraints of the form Hx <= b with time dependency.
    
    """
    def __init__(self, H : np.ndarray , b : np.ndarray, start:float, end:float , is_hard : bool =True, penalty_weight : float | None = None):
        
        
        super().__init__(H, b, is_hard, penalty_weight)
        self.start = start
        self.end   = end

        if self.start > self.end: # the equal condition is accepted (corresponding to a singleton)
            raise ValueError("start must be less than end.")
        

class MPCProblem:

    """
    Base class to define all the parameters for the MPC controller.
    """

    def __init__(self, system               : ContinuousLinearSystem| MultiAgentSystem, 
                       horizon              : int,
                       Q                    : np.ndarray,
                       R                    : np.ndarray, 
                       QT                   : np.ndarray | None = None,
                       global_penalty_weight: float             = 1.0, 
                       solver               : str |None         = None, 
                       slack_penalty        : str               = "SQUARE"):

        """
        Initializes the MPCParameters with the given parameters.

        :param system: The state-space model of the system. The state space model should be described in desceret time.
        :type system: ContinuousLinearSystem| MultiAgentSystem
        :param horizon: The prediction horizon for the MPC.
        :type horizon: int
        :param Q: The state weighting matrix.
        :type Q: numpy.ndarray
        :param R: The input weighting matrix.
        :type R: numpy.ndarray
        :param QT: The terminal state weighting matrix.
        :type QT: numpy.ndarray | None
        :param global_penalty_weight: The global penalty weight for the cost function.
        :type global_penalty_weight: float
        :param solver: The solver to be used for optimization.
        :type solver: str |None
        :param slack_penalty: The type of penalty for slack variables, default is 'SQUARE'.
        :type slack_penalty: str
        :param u_constraints: List of input constraints.
        :type u_constraints: list[Constraint]
        :param x_constraints: List of state constraints.
        :type x_constraints: list[Constraint]
        :param y_constraints: List of output constraints.
        :type y_constraints: list[Constraint]
        :param terminal_constraints: List of terminal constraints.
        :type terminal_constraints: list[Constraint]
        :param terminal_set: The terminal set for the MPC.
        :type terminal_set: Polytope | None
        :param dual_mode_controller: The dual mode controller for the MPC.
        :type dual_mode_controller: numpy.ndarray
        :param dual_mode_horizon: The horizon for the dual mode.
        :type dual_mode_horizon: int
        :param reference_controller: The reference controller for the MPC.
        :type reference_controller: numpy.ndarray
        :param soft_tracking: Whether soft tracking is enabled.
        :type soft_tracking: bool
        :param tracking_penalty_weight: The penalty weight for soft tracking.
        :type tracking_penalty_weight: float
        :param reference_reached_at_steady_state: Whether the reference is reached at steady state.
        :type reference_reached_at_steady_state: bool
        :param dt: The time step for the system.
        :type dt: float
        """
        
        # Store the state-space model
        self.system = system
        self.dt     = system.dt

        # MPC parameters
        self.horizon = horizon
        self.Q       = Q
        self.R       = R
        self.QT      = np.zeros((system.state_dim, system.state_dim)) if QT is None else QT

        self.global_penalty_weight = global_penalty_weight
        self.solver                = solver #! we need to check which solvers are available

        self.slack_penalty         = slack_penalty  # Changed from slack_norm to slack_penalty

        # Constraints as lists of `Constraint` objects
        self.u_constraints          : list[Constraint] = []
        self.x_constraints          : list[Constraint] = []
        self.y_constraints          : list[Constraint] = []
        self.terminal_constraints   : list[Constraint] = []
        self.terminal_set           : Constraint | None  = None

        # Dual mode parameters
        self.dual_mode_controller : np.ndarray = np.zeros((self.system.input_dim, self.system.state_dim))
        self.dual_mode_horizon    : int        = 0
        self.reference_controller : np.ndarray = np.zeros((self.system.input_dim, self.system.state_dim)) # This is the controller that creates the reference input to the system as U_ref = -L_ref*X


        # Tracking parameters
        self.soft_tracking           = False
        self.tracking_penalty_weight = 1.  # Default penalty weight for soft tracking
        self.reference_reached_at_steady_state = True
        
        # Advanced
        self.state_time_constraints : list[TimedConstraint] = []
    
    def add_input_magnitude_constraint(self, limit :float , input_index : int | None = None, is_hard : bool = True, penalty_weight : float =1.):
        """
        Add input magnitude constraint: -limit <= u <= limit.
        
        :param limit: The magnitude limit for the input.
        :type limit: float
        :param input_index: The index of the input to apply the constraint to. If None, apply to all inputs.
        :type input_index: int | None
        :param is_hard: Whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for soft constraints.
        :type penalty_weight: float
        """

        limit = float(limit)
        input_index = int(input_index) if input_index is not None else None
        penalty_weight = float(penalty_weight)

        if penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative.")


        if input_index is None:
            A = np.vstack([np.eye(self.system.input_dim), -np.eye(self.system.input_dim)])
            b = np.array([limit] * self.system.input_dim * 2)
        else:
            A = np.array([[1 if i == input_index else 0 for i in range(self.system.input_dim)]])
            A = np.vstack([A, -A])
            b = np.array([limit, limit])
        
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.u_constraints.append(constraint)

    def add_input_bound_constraint(self, limits : tuple | float, input_index : int | None = None, is_hard : bool =True, penalty_weight : float = 1.):
        """
        Adds input bounds constraints. The constraints can be symmetric or asymmetric based on `limits`.

        :param limits: If a single float, the bounds are symmetric: -limits <= u_t <= limits.
                          If a tuple (lb, ub), the bounds are asymmetric: lb <= u_t <= ub.
        :type limits: float or tuple
        :param input_index: If None, apply the constraints to all inputs uniformly.
                           If an int, apply the constraint to a specific input.
        :type input_index: int | None
        :param is_hard: Whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for soft constraints.
        :type penalty_weight: float
        """

        # Determine the bounds
        if isinstance(limits, (int, float)):
            lb, ub = -limits, limits  # Symmetric bounds
        elif isinstance(limits, tuple) and len(limits) == 2:
            lb, ub = limits  # Asymmetric bounds
        else:
            raise ValueError("limits must be a single value or a tuple of two values (lb, ub).")

        if input_index is None:
            # Apply uniformly to all inputs
            n_inputs = self.system.input_dim
            A = np.vstack([np.eye(n_inputs), -np.eye(n_inputs)])  # Identity and negative identity for constraints
            b = np.hstack([ub * np.ones(n_inputs), -lb * np.ones(n_inputs)])
        else:
            # Apply to a specific input
            A = np.zeros((2, self.system.input_dim))
            A[0, input_index] = 1  # Upper bound for the specified input
            A[1, input_index] = -1  # Lower bound for the specified input
            b = np.array([ub, -lb])

        # Create and store the constraint
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.u_constraints.append(constraint)
        
        

    def add_state_magnitude_constraint(self, limit :float , state_index  : int | None = None, is_hard : bool = True, penalty_weight : float =1.):
        """Add state magnitude constraint: -limit <= x[state_index] <= limit."""
        
        if state_index is None:
            A = np.vstack([np.eye(self.system.state_dim), -np.eye(self.system.state_dim)])
            b = np.array([limit] * self.system.state_dim * 2)
        else:
            A = np.array([[1 if i == state_index else 0 for i in range(self.system.state_dim)]])
            A = np.vstack([A, -A])
            b = np.array([limit, limit])
        
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)


    def add_state_bound_constraint(self,  limits : tuple | float, state_index  : int | None = None, is_hard : bool =True, penalty_weight : float = 1.):
        """
        Adds state bounds constraints. The constraints can be symmetric or asymmetric based on `limits`.
        
        :param limits: If a single float, the bounds are symmetric: -limits <= x_t <= limits.
                            If a tuple (lb, ub), the bounds are asymmetric: lb <= x_t <= ub.
        :type limits: float or tuple
        :param state_index: If None, apply the constraints to all states uniformly.
                            If an int, apply the constraint to a specific state.
        :type state_index: int | None
        :param is_hard: Whether the constraint is hard or soft.
        :type is_hard: bool
        :param penalty_weight: The penalty weight for soft constraints.
        :type penalty_weight: float
        """
        # Determine the bounds
        if isinstance(limits, (int, float)):
            lb, ub = -limits, limits  # Symmetric bounds
        elif isinstance(limits, tuple) and len(limits) == 2:
            lb, ub = limits  # Asymmetric bounds
        else:
            raise ValueError("limits must be a single value or a tuple of two values (lb, ub).")

        if state_index is None:
            # Apply uniformly to all states
            n_states = self.system.state_dim
            A = np.vstack([np.eye(n_states), -np.eye(n_states)])  # Identity and negative identity for constraints
            b = np.hstack([ub * np.ones(n_states), -lb * np.ones(n_states)])
        else:
            # Apply to a specific state
            A = np.zeros((2, self.system.state_dim))  # Number of columns matches state dimension
            A[0, state_index] = 1  # Upper bound for the specified state
            A[1, state_index] = -1  # Lower bound for the specified state
            b = np.array([ub, -lb])

        # Create and store the constraint
        constraint = Constraint(A, b, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)
        
        
    def add_general_state_constraints(self, Hx : np.ndarray, bx : np.ndarray, is_hard : bool =True, penalty_weight : int=1):
        """Add general state constraints of the form Hx * x <= bx."""


        if Hx.shape[1] != self.system.state_dim:
            raise ValueError("The number of columns in A must match the state dimension of the system.")
        

        constraint = Constraint(Hx, bx, is_hard=is_hard, penalty_weight=penalty_weight)
        self.x_constraints.append(constraint)


    def add_general_input_constraints(self, Hu : np.ndarray, bu : np.ndarray, is_hard : bool =True, penalty_weight : int=1):
        """Add general input constraints of the form Hu * u <= bu."""
        
        if Hu.shape[1] != self.system.input_dim:
            raise ValueError("The number of columns in A must match the input dimension of the system.")
        
        constraint = Constraint(Hu, bu, is_hard=is_hard, penalty_weight=penalty_weight)
        self.u_constraints.append(constraint)

    def add_general_state_time_constraints(self, Hx : np.ndarray, bx : np.ndarray, start_time:float, end_time:float,is_hard : bool =True, penalty_weight : int=1):
        """Add general time state constraints of the form A[x,t] <= bx."""
        
        if Hx.shape[1] != self.system.state_dim+1:
            raise ValueError("The number of columns in A must match the state dimension of the system + 1 to account for the time dimension.")
        
        constraint = TimedConstraint(Hx, bx, start_time, end_time, is_hard=is_hard, penalty_weight=penalty_weight)
        self.state_time_constraints.append(constraint)

    
    def reach_refererence_at_steady_state(self, option :bool = True):
        """
        Set the option to reach the reference at steady state or let the reference be reached from any state.
        
        :param option: If True, the reference is reached at steady state. If False, the reference can be reached from any state.
        :type option: bool
        """

        self.reference_reached_at_steady_state = option


        
    def soften_tracking_constraint(self, penalty_weight : float = 1.):
        """Enable softening of the tracking constraint with the specified penalty weight."""

        if penalty_weight < 0.:
            raise ValueError("penalty_weight must be non-negative.")
        
        self.soft_tracking           = True
        self.tracking_penalty_weight = penalty_weight
        
    
    def __str__(self):

        str_out = ""
        str_out += f"System: {self.system}\n"
        str_out += f"Time Step: {self.dt}\n"
        str_out += f"System type: {type(self.system)}\n"
        str_out += f"Horizon: {self.horizon}\n"
        str_out += f"Q: {self.Q}\n"
        str_out += f"R: {self.R}\n"
        str_out += f"QT: {self.QT}\n"
        str_out += f"Global Penalty Weight: {self.global_penalty_weight}\n"
        str_out += f"Solver: {self.solver}\n"
        str_out += f"Slack Penalty: {self.slack_penalty}\n"
        str_out += f"dual mode horizon: {self.dual_mode_horizon} (0 means deactivated)\n"
        str_out += f"dual mode controller: {self.dual_mode_controller}\n"
        str_out += f"Reference controller: {self.reference_controller}\n"
        str_out += f"Soft tracking: {self.soft_tracking}\n"
        str_out += f"Tracking Penalty Weight: {self.tracking_penalty_weight}\n"
        
        return str_out


class TimedMPC:
    r"""
    MPC controller class for set-point tracking of linear systems.


    Main controller :

    .. math::

        &\min_{x,u} sum_{t=0}^{N-1} (x_t - x_{ref})^T Q (x_t - x_{ref}) + (u_t - u_{ref})^T R (u_t - u_{ref})\\
        &s.t.\\
        &x_{t+1} = A x_t + B u_t + B_d d_t\\
        &H_u u_t \leq h_u \\
        &H_x x_t \leq h_x \\
        &H_y (C x_t + D u_t) \leq h_y \\
        &x_0 = x_0
    """

    def __init__(self, mpc_params : MPCProblem):

        """
        Initializes the TrackingMPC with the given MPC parameters.

        :param mpc_params: The parameters for the MPC.
        :type mpc_params: MPCParameters
        """

        # Extract MPC parameters
        self.params               : MPCProblem   = mpc_params
        self.system               : ContinuousLinearSystem | MultiAgentSystem = self.params.system
        self.A, self.B          = self.system.c2d()


        self.n                    : int = self.system.state_dim
        self.m                    : int = self.system.input_dim
        self.N                    : int = self.params.horizon
        self.Q                    : np.ndarray = self.params.Q
        self.R                    : np.ndarray = self.params.R
        self.QT                   : np.ndarray = self.params.QT
        
        self.u_constraints         : list[Constraint] = self.params.u_constraints
        self.x_constraints         : list[Constraint] = self.params.x_constraints
        self.y_constraints         : list[Constraint] = self.params.y_constraints
        self.terminal_constraints  : Constraint       = self.params.terminal_constraints
        self.global_penalty_weight : float            = self.params.global_penalty_weight
        
        self.solver               : str        = self.params.solver
        self.slack_penalty        : float      = self.params.slack_penalty
        self.terminal_set         : Constraint = self.params.terminal_set
        self.dual_mode_controller : np.ndarray = self.params.dual_mode_controller
        self.dual_mode_horizon    : int        = self.params.dual_mode_horizon
        
        # Define decision variables
        self.x     = cp.Variable((self.n, self.N + 1))
        self.u     = cp.Variable((self.m, self.N))
        self.slack = cp.Variable((1, self.N + 1), nonneg=True)  # Slack variable for soft constraints
        self.x_ref = cp.Parameter(self.n)
        self.x0    = cp.Parameter(self.n)                 # Initial state parameter

        # Option for soft tracking constraint from mpc_params
        self.soft_tracking           = self.params.soft_tracking
        self.tracking_penalty_weight = self.params.tracking_penalty_weight

        self.time_state_constraints          : list[TimedConstraint] = mpc_params.state_time_constraints
        self.time_step                       : float                 = mpc_params.system.dt
        self.time_par                        : cp.Parameter          = cp.Parameter(1)  # Time parameter for the time state constraints


        self.activation_parameters : dict[Constraint,cp.Parameter] = {}
        for constraint in self.time_state_constraints:
            self.activation_parameters[constraint] = cp.Parameter(shape=(mpc_params.horizon+1,),value =np.ones(mpc_params.horizon+1,),integer = True)
               

        self._setup_problem()  # Call the parent class's setup method to initialize the problem
        
    def set_time(self, time : float):
        """
        Set the current time for the timed constraints.

        :param time: The current time.
        :type time: float
        """
        
        self.time_par.value = np.array(time).reshape(1,)

    def set_reference(self, reference : np.ndarray):
        """
        Set the reference state for the MPC.

        :param reference: The reference state.
        :type reference: np.ndarray
        """
        
        if reference.shape != (self.n,):
            raise ValueError(f"Reference must have shape ({self.n},), but got {reference.shape}.")
        self.x_ref.value = reference

    def set_initial_state(self, x0 : np.ndarray):
        """
        Set the initial state for the MPC.

        :param x0: The initial state.
        :type x0: np.ndarray
        """
        
        if x0.shape != (self.n,):
            raise ValueError(f"Initial state must have shape ({self.n},), but got {x0.shape}.")
        self.x0.value = x0

    def update_activation_parameters(self, time : float):
        """
        Update the activation parameters for the timed constraints.

        :param time: The current time.
        :type time: float
        """
        
        time_horizon_range = np.linspace(time , self.time_step*(self.N) + time ,self.N+1)
        for constraint in self.time_state_constraints:
            print("constraint active from ",constraint.start," to ",constraint.end)
            print("time horizon range: ",time_horizon_range)
            print("time parameter value: ",self.time_par.value)

            active_set = np.bitwise_and(time_horizon_range >= constraint.start, time_horizon_range <= constraint.end)
            print("active set: ",active_set)
            self.activation_parameters[constraint].value = active_set
    

    def _setup_problem(self):

        self.cost = 0
        self.constraints = [self.x[:, 0] == self.x0]

        # Main MPC horizon loop
        slack_variables = []  # To collect slack variables for other soft constraints
        for t in range(self.N):
            # Add the tracking cost
            self.cost += cp.quad_form(self.x[:, t] - self.x_ref, self.Q) + cp.quad_form(self.u[:, t], self.R)

            # System dynamics including disturbance if it exists
            self.constraints += [self.x[:, t + 1] == self.A @self.x[:, t] + self.B @ self.u[:, t]]

            # Add input constraints
            for constraint in self.u_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.u[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.u[:, t] <= b + np.ones(H.shape[0]) * slack]

            # Add state constraints
            for constraint in self.x_constraints:
                H,b = constraint.to_polytope()
                if constraint.is_hard:
                    self.constraints += [H @self.x[:, t] <= b]
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @self.x[:, t] <= b + np.ones(H.shape[0]) * slack]

            # add timed constraints 
            time = self.time_par + t * self.time_step
            # state time constraints (Only difference with standard set point tracking MPC)
            for constraint in self.time_state_constraints:
                H,b = constraint.to_polytope()
                activation_bit   = self.activation_parameters[constraint][t]
                if constraint.is_hard:
                    self.constraints += [H @ cp.hstack((self.x[:, t], time)) <= b + (1 - activation_bit) * 1e6 ] # add time varying constraints
                else:
                    slack = cp.Variable(nonneg=True)
                    slack_variables.append((slack, constraint.penalty_weight))
                    self.constraints += [H @ cp.hstack((self.x[:, t], time)) <= b + (1 - activation_bit) * 1e6 + np.ones(H.shape[0]) * slack] # add time varying constraints

        time = self.time_par + (self.N) * self.time_step
        # state time constraints (Only difference with standard set point tracking MPC)
        for constraint in self.time_state_constraints:
            H,b = constraint.to_polytope()
            activation_bit   = self.activation_parameters[constraint][self.N]
            if constraint.is_hard:
                self.constraints += [H @ cp.hstack((self.x[:, self.N], time)) <= b + (1 - activation_bit) * 1e6 ] # add time varying constraints
            else:
                slack = cp.Variable(nonneg=True)
                slack_variables.append((slack, constraint.penalty_weight))
                self.constraints += [H @ cp.hstack((self.x[:, self.N], time)) <= b + (1 - activation_bit) * 1e6 + np.ones(H.shape[0]) * slack] # add time varying constraints

        # Terminal cost
        self.cost += cp.quad_form(self.x[:, self.N] - self.x_ref, self.QT)
        
        # Add slack penalties for other soft constraints
        for slack, penalty_weight in slack_variables:
            if self.slack_penalty == 'LINEAR':
                self.cost += self.global_penalty_weight * penalty_weight * slack  # Linear penalty
            elif self.slack_penalty == 'SQUARE':
                self.cost += self.global_penalty_weight * penalty_weight * cp.square(slack)  # Quadratic penalty
            else :
                raise ValueError("Invalid slack penalty type. Must be 'LINEAR' or 'SQUARE'.")
        
        # Create the problem instance
        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        

    def get_control_action(self, x0 : np.ndarray, t0 : float, reference : np.ndarray)-> np.ndarray:
        """Compute the control action for a given state, reference, and optional disturbance."""

        _, u_pred = self.compute(x0, t0, reference)

        return np.atleast_1d(u_pred[:, 0])  # Return the entire vector for the first control input

    
    def get_state_and_control_trajectory(self, x0 : np.ndarray, t0:float , reference : np.ndarray)-> tuple[np.ndarray, np.ndarray,float]:
        
        x_pred, u_pred = self.compute(x0, t0, reference)
        terminal_time = self.time_par.value + self.time_step * self.N
        return x_pred, u_pred, terminal_time
    
    def compute(self,  x0 : np.ndarray, t0:float , reference : np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        # set up problem parameters
        self.set_reference(reference)
        self.set_time(t0)
        self.set_initial_state(x0)
        self.update_activation_parameters(t0) # selects active constraint

        self.problem.solve(solver=self.solver)

        if self.problem.status != cp.OPTIMAL:
            raise ValueError(f"MPC problem is {self.problem.status}.")
        return self.x.value, self.u.value
