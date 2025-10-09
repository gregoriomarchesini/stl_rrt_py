import casadi as ca
import numpy  as np

from ..stl.linear_system       import MultiAgentSystem, ContinuousLinearSystem
from ..stl.logic               import Formula,is_dnf
from ..stl.parameter_optimizer import TimeVaryingConstraint




class NashController:
    
    def __init__(self, multi_agent_system : MultiAgentSystem, 
                       task_dict          : dict[Formula, list[TimeVaryingConstraint]],
                       horizon            : int,
                       dt                 : float,
                       verbose            : bool = False,
                       solver             : str = "sqpmethod"):


        self.task_dict        : dict[Formula, list[TimeVaryingConstraint]] = task_dict
        self.mas              : MultiAgentSystem                           = multi_agent_system
        self.task_activations : dict[Formula, bool]                        = {task:False for task in task_dict.keys()}
        self.horizon          : int                                        = horizon
        self.dt               : float                                      = dt
        self.verbose          : bool                                       = verbose
        self.solver           : str                                        = solver

        solvers = ["ipopt","sqpmethod"]
        if solver not in solvers:
            raise ValueError(f"Solver {solver} not recognized. Available solvers are {solvers}.")
        

        self.A, self.B = self.mas.c2d()

        self.optimizer = ca.Opti()

        self.x0       = self.optimizer.parameter(self.mas.state_dim)
        self.time_par = self.optimizer.parameter(1,)
        self.time_activation_para_dict = dict()
        for task, constraints in self.task_dict.items():
            for constraint in constraints:
                self.time_activation_para_dict[constraint] = self.optimizer.parameter(self.horizon+1,)
       
        self.task_activation_params = { task : self.optimizer.parameter(1) for task in self.task_dict.keys() }

        self.agent_state_trj = { system.name : self.optimizer.variable(system.state_dim, self.horizon+1) for system in self.mas }
        self.agent_input_trj = { system.name : self.optimizer.variable(system.input_dim, self.horizon) for system in self.mas }
        self.agents_pos_trj  = {}
        
        for system in self.mas:
            try:
                pos_dims = system.get_dims_from_state_name('position')
            except Exception:
                try:
                    pos_dims  = system.get_dims_from_state_name('pos')
                except Exception:
                    raise ValueError(f"System {system.name} does not have a 'position' or 'pos' state. The nash controller expects each agent to have a position state for collsion avoidance")
        
            self.agents_pos_trj[system.name] = self.agent_state_trj[system.name][pos_dims, :]


        self.setup()

    def setup(self):

        self.cost = 0

        # Main MPC horizon loop
        slack_variables = []  # To collect slack variables for other soft constraints
        
        for system in self.mas:
            x_system = self.agent_state_trj[system.name]
            u_system = self.agent_input_trj[system.name]
            self.cost = ca.sumsqr(u_system)

            A_system, B_system = system.c2d(self.dt)
            
            for t in range(self.horizon):

                self.constraints += [x_system[:, t + 1] == A_system @x_system[:, t] + B_system @ u_system[:, t]]

                # add timed constraints 
                time = self.time_par + t * self.dt
                # state time constraints (Only difference with standard set point tracking MPC)
                
                
                for task, constraint in self.time_state_constraints:
                    H,b = constraint.to_polytope()
                    activation_bit   = self.activation_parameters[constraint][t]
                    if constraint.is_hard:
                        self.constraints += [H @ cp.hstack((self.x[:, t], time)) <= b + (1 - activation_bit) * 1e6 ] # add time varying constraints
                    else:
                        slack = cp.Variable(nonneg=True)
                        slack_variables.append((slack, constraint.penalty_weight))
                        self.constraints += [H @ cp.hstack((self.x[:, t], time)) <= b + (1 - activation_bit) * 1e6 + np.ones(H.shape[0]) * slack] # add time varying constraints

        
        # add global constraints on the system
        for t in range(self.horizon):
        
        
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