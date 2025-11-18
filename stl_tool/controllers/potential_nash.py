import casadi as ca
import numpy  as np

from ..stl.linear_system       import MultiAgentSystem, ContinuousLinearSystem
from ..stl.parameter_optimizer import TimeVaryingConstraint
from .analytical_qp            import AnalyticalQp


class PotentialNashController:
    
    def __init__(self, multi_agent_system,
                       task_constraints,
                       horizon,
                       dt,
                       verbose: bool = False,
                       solver: str = "ipopt",
                       kappa_gain: float = 1.0,
                       bigM: float = 1e3,
                       slack_penalty: float = 1e5):

        # Public inputs
        self.task_constraints = task_constraints
        self.mas = multi_agent_system
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.verbose = bool(verbose)
        self.solver = solver

        # tuning
        self.bigM = float(bigM)
        self.slack_penalty = float(slack_penalty)

        # fast analytical controller used for warm-start guess
        self.fast_controller = AnalyticalQp(system=multi_agent_system,
                                           task_constraints=task_constraints,
                                           kappa=kappa_gain,
                                           add_collision_gradient=True)

        solvers = ["ipopt", "sqpmethod"]
        if solver not in solvers:
            raise ValueError(f"Solver {solver} not recognized. Available solvers are {solvers}.")

        # System-level discrete dynamics (stacked A,B for the full state)
        self.A, self.B = self.mas.c2d()

        # CasADi Opti
        self.optimizer = ca.Opti()
        self.x0 = self.optimizer.parameter(self.mas.state_dim)   # initial state parameter (full stacked)
        self.time_par = self.optimizer.parameter(1,)             # scalar time parameter

        # activation parameters and slack variables per constraint
        self.time_activation_param_dict = dict()
        self.slack_variables_dict = dict()
        for constraint in self.task_constraints:
            # activation parameter for timesteps 0..horizon (inclusive) -> length horizon+1
            self.time_activation_param_dict[constraint] = self.optimizer.parameter(self.horizon + 1,)
            # slack variables: rows = number of inequality rows in constraint.b, cols = horizon+1
            m = len(constraint.b)
            self.slack_variables_dict[constraint] = self.optimizer.variable(m, self.horizon + 1)

        # per-agent trajectories (Opti variables)
        self.agent_state_trj = {system.name: self.optimizer.variable(system.state_dim, self.horizon + 1)
                                for system in self.mas}
        self.agent_input_trj = {system.name: self.optimizer.variable(system.input_dim, self.horizon)
                                for system in self.mas}

        # stacked trajectories (vertical concatenation of the same Opti variables above)
        self.global_state_trj = ca.vcat([self.agent_state_trj[system.name] for system in self.mas])
        self.global_input_trj = ca.vcat([self.agent_input_trj[system.name] for system in self.mas])

        # positions for collision cost (view into each agent state traj)
        self.agents_pos_trj = {}
        for system in self.mas:
            try:
                pos_dims = system.get_dims_from_state_name('position')
            except Exception:
                try:
                    pos_dims = system.get_dims_from_state_name('pos')
                except Exception:
                    raise ValueError(f"System {system.name} does not have a 'position' or 'pos' state.")
            self.agents_pos_trj[system.name] = self.agent_state_trj[system.name][pos_dims, :]

        # warm-start guesses
        self.agent_state_trj_guess = {system.name: np.zeros((system.state_dim, self.horizon + 1))
                                      for system in self.mas}
        self.agent_input_trj_guess = {system.name: np.zeros((system.input_dim, self.horizon))
                                      for system in self.mas}
        self.is_warm_start_available = False

        # cost and constraints holders (will be populated by setup)
        self.cost = 0
        self.constraints = []

        # build the problem
        self.setup()

    def setup(self):
        """Builds the Opti problem (dynamics, time-varying constraints, cost, solver options)."""

        # -------------------------
        # Dynamics constraints (per-agent)
        # -------------------------
        for system in self.mas:
            x_system = self.agent_state_trj[system.name]
            u_system = self.agent_input_trj[system.name]
            A_sys, B_sys = system.c2d()
            for t in range(self.horizon):
                self.constraints += [x_system[:, t + 1] == A_sys @ x_system[:, t] + B_sys @ u_system[:, t]]

        # -------------------------
        # Time-varying linear constraints (apply for t = 0..horizon)
        # -------------------------
        for t in range(self.horizon + 1):
            time = self.time_par + t * self.dt
            x = self.global_state_trj[:, t]   # full stacked state at time t
            for constraint in self.task_constraints:
                # activation param and slack at time t
                activation_bit = self.time_activation_param_dict[constraint][t]
                slack = self.slack_variables_dict[constraint][:, t]
                H, b = constraint.H, constraint.b

                # shape checks (catch mismatches early)
                assert H.shape[0] == slack.shape[0], (
                    f"Slack rows {slack.shape[0]} must match H rows {H.shape[0]} for constraint {constraint}"
                )

                # inequality: H * [x; time] <= b + (1-activation)*bigM + slack, slack >= 0
                stacked = ca.vcat((x, time))
                self.constraints += [H @ stacked <= b + (1.0 - activation_bit) * self.bigM + slack]
                self.constraints += [slack >= 0]

        # -------------------------
        # Cost
        # -------------------------
        # collision cost: evaluate for states t=0..horizon (include final state collision penalization)
        for t in range(self.horizon + 1):
            # collision avoidance cost (Gaussian) — only valid if position at this t exists (we built pos arrays with horizon+1 cols)
            for i, sys_i in enumerate(self.mas):
                pos_i = self.agents_pos_trj[sys_i.name][:, t]
                for j, sys_j in enumerate(self.mas):
                    if i <= j:
                        continue
                    pos_j = self.agents_pos_trj[sys_j.name][:, t]
                    dist_ij_square = ca.sumsqr(pos_i - pos_j)
                    variance = 1.0
                    scaling = 100
                    # fixed parentheses for correct Gaussian exponent
                    gaussian = scaling * (1.0 / ca.sqrt(2 * ca.pi * variance)) * ca.exp(-dist_ij_square / (2 * variance**2))
                    self.cost += gaussian

        # input cost: for t = 0..horizon-1 (inputs only defined up to horizon-1)
        for t in range(self.horizon):
            u = self.global_input_trj[:, t]
            self.cost += ca.sumsqr(u) *100.

        # slack variable cost: penalize slacks at all timesteps 0..horizon
        for t in range(self.horizon + 1):
            for constraint in self.task_constraints:
                slack = self.slack_variables_dict[constraint][:, t]
                self.cost += ca.sumsqr(slack) * self.slack_penalty

        # initial state constraint
        self.constraints += [self.global_state_trj[:, 0] == self.x0]

        # -------------------------
        # finalize: objective + constraints to Optimizer
        # -------------------------
        self.optimizer.minimize(self.cost)
        self.optimizer.subject_to(self.constraints)

        # solver options
        if self.solver == "ipopt":
            opts = {'print_time': 0, 'ipopt': {'print_level': 0}}
            self.optimizer.solver(self.solver, opts)

        elif self.solver == "sqpmethod":
            p_opts = {
                "qpsol": "osqp",
                "max_iter": 150,
                "hessian_approximation": "exact",
                "convexify_strategy": "regularize",
                "convexify_margin": 1e-3,
                "init_feasible": False,
                "elastic_mode": True,
                "second_order_corrections": True,
                "c1": 1e-2,
                "beta": 0.9,
                "tol_du": 2.5e-2,
                "tol_pr": 1e-4,
                'gamma_0': 0.1,
                "print_iteration": True,
                "qpsol_options.error_on_fail": 0,
                "qpsol_options.print_time": 0,
                "qpsol_options.verbose": 0,
                "qpsol_options.warm_start_primal": False,
                "qpsol_options.osqp.verbose": False,
                "qpsol_options.osqp.eps_abs": 1e-5,
                "qpsol_options.osqp.eps_rel": 1e-5,
                "qpsol_options.osqp.polish": True,
            }
            self.optimizer.solver(self.solver, p_opts)

    def get_input(self, x_0: np.ndarray, t_0: float):
        """Set parameters, warm-start, solve and return the first control input (stacked)."""

        # set initial full-state and time parameter
        self.optimizer.set_value(self.x0, x_0)
        self.optimizer.set_value(self.time_par, t_0)

        # set initial trajectory guess (warm start)
        if self.is_warm_start_available:
            for system in self.mas:
                self.optimizer.set_initial(self.agent_state_trj[system.name],
                                           self.agent_state_trj_guess[system.name])
                self.optimizer.set_initial(self.agent_input_trj[system.name],
                                           self.agent_input_trj_guess[system.name])
        else:
            # rollout the fast controller to populate guesses
            x_rol = x_0.copy()
            for t_h in range(self.horizon):
                u_fast = self.fast_controller.get_input(x_rol, t_0 + t_h * self.dt)
                # distribute to each agent
                for system in self.mas:
                    idxs = self.mas.get_system_state_dims(system.name)
                    x_0_system = x_rol[idxs]
                    self.agent_state_trj_guess[system.name][:, t_h] = x_0_system
                    u_dims = self.mas.get_system_input_dims(system.name)
                    u_system = u_fast[u_dims]
                    if t_h < self.horizon - 1:
                        self.agent_input_trj_guess[system.name][:, t_h] = u_system
                
                # advance rollout with stacked A,B
                x_rol = self.A @ x_rol + self.B @ u_fast

            # fill last column of state guess with the last rolled state
            for system in self.mas:
                # last state guess col is already zero for inputs and was set for states upto horizon-1:
                self.agent_state_trj_guess[system.name][:, -1] = self.agent_state_trj_guess[system.name][:, -2]

            self.is_warm_start_available = True

            # push the initial guesses into the opti object
            for system in self.mas:
                self.optimizer.set_initial(self.agent_state_trj[system.name],
                                           self.agent_state_trj_guess[system.name])
                self.optimizer.set_initial(self.agent_input_trj[system.name],
                                           self.agent_input_trj_guess[system.name])

        # set activation parameter values consistently for 0..horizon
        for constraint in self.task_constraints:
            time_activation_param = self.time_activation_param_dict[constraint]
            for tau in range(self.horizon + 1):
                time = t_0 + tau * self.dt
                val = 1 if constraint.is_active(time) else 0
                self.optimizer.set_value(time_activation_param[tau], val)
        
        
        # solve
        try:
            sol = self.optimizer.solve()
        except RuntimeError:
            print("Solver failed!")
            return None

        # read solution using sol.value(...) everywhere (important)
        u_val = sol.value(self.global_input_trj)   # stacked inputs shape (total_input_dim, horizon)
        x_val = sol.value(self.global_state_trj)   # stacked states shape (total_state_dim, horizon+1)

    
        # Save guesses (IMPORTANT: use sol.value, not optimizer.value)
        for system in self.mas:
            u_val_system = sol.value(self.agent_input_trj[system.name])
            x_val_system = sol.value(self.agent_state_trj[system.name])

            # shift by one: new guess = solution shifted, and last column zero/last state repeated
            u_guess = np.hstack([u_val_system[:, 1:], np.zeros((system.input_dim, 1))])
            x_guess = np.hstack([x_val_system[:, 1:], x_val_system[:, -1][:, np.newaxis]])

            self.agent_state_trj_guess[system.name] = x_guess
            self.agent_input_trj_guess[system.name] = u_guess

        # return first control for the full stacked system
        return u_val[:, 0]