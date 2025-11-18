from stl_tool.stl import ContinuousLinearSystem,TimeVaryingConstraint, MultiAgentSystem
import numpy  as np


class AnalyticalQp:

    """ 
    Analytical QP controller with time-varying barrier functions and optional collision avoidance. 

    Time-varying linear constraints are used to represent STL tasks in the form of space-time inequalities 
    H * [x; t] <= b
    Which can be read row-wise as a set of linear barrier functions
    -h_i^T * x - h_t * t +  b_i \geq 0.

    Based on this representation, an analytical controller that keeps the system within the constraints can be defined using the smooth minimu approximation of all the 
    barrier functions.
    """

    def __init__(self, system                 : ContinuousLinearSystem | MultiAgentSystem , 
                       task_constraints       : list[TimeVaryingConstraint], 
                       kappa                  : float = 1.0, 
                       add_collision_gradient : bool  = False):

        self.system                   : ContinuousLinearSystem | MultiAgentSystem = system
        self.task_constraints         : list[TimeVaryingConstraint]               = task_constraints
        self.dt                       : float                                     = system.dt
        self.eta                      : float                                     = 100
        self.kappa                    : float                                     = kappa
        self.collision_gradient       : bool                                      = add_collision_gradient
        self.B_pinv                   : np.ndarray                                 = np.linalg.pinv(system.B)

        if self.collision_gradient and not isinstance(system, MultiAgentSystem):
            raise ValueError("Collision gradient can only be added for multi-agent systems.")
        for sys in system.systems if isinstance(system, MultiAgentSystem) else [system]:
            if 'position' not in sys.state_naming and 'pos' not in sys.state_naming:
                raise ValueError(f"System {sys.name} does not have a 'position' or 'pos' state. The analytical QP controller expects each agent to have a position state for collision avoidance activatted")

    def get_input(self, x_0: np.ndarray, t_0: float):

        # --- 1. Gather active constraints for the barrier
        active_constraints = []
        for c in self.task_constraints:
            if c.start_time <= t_0 <= c.end_time:
                for ii in range(c.H.shape[0]):
                    H_row = c.H[ii, :]
                    h_val = -H_row @ np.hstack([x_0, t_0]) + c.b[ii]
                    active_constraints.append((H_row, h_val))

        if not active_constraints:
            return np.zeros((self.system.B.shape[1],))

        # --- 2. Stable log-sum-exp trick
        h_vals = np.array([h for _, h in active_constraints])
        h_min = np.min(h_vals)
        exp_terms = np.exp(-self.eta * (h_vals - h_min))
        denominator = np.sum(exp_terms)

        grad_x = np.zeros_like(x_0)
        grad_t = 0.0
        for (H_row, h_val), exp_term in zip(active_constraints, exp_terms):
            grad_x += exp_term * (-H_row[:-1])
            grad_t += exp_term * (-H_row[-1])
        grad_x /= denominator
        grad_t /= denominator
        barrier = h_min - (1.0 / self.eta) * np.log(denominator)

        # --- 3. Compute repulsive force for collision avoidance
        f_x = np.zeros_like(x_0)

        if self.collision_gradient:
            kkappa = 1.0
            d_safe = 3.0  # safety distance (tunable)

            for sys in self.system.systems:
                try:
                    dim = self.system.get_system_state_dims(sys.name, 'position')
                except Exception:
                    dim = sys.get_dims_from_state_name(sys.name,"pos")

                pos = x_0[dim]
                total_repulse = np.zeros_like(pos)

                for other_sys in self.system.systems:
                    if other_sys.name == sys.name:
                        continue

                    try:
                       other_dim = self.system.get_system_state_dims(other_sys.name, 'position')
                    except Exception:
                        other_dim = other_sys.get_dims_from_state_name(other_sys.name,"pos")

                    other_pos = x_0[other_dim]
                    pos_diff = pos - other_pos  # <-- flip direction (points away)
                    dist = np.linalg.norm(pos_diff)

                    if dist < 1e-6:
                        continue  # skip zero distance

                    if dist < d_safe:
                        # Repulsive force (inverse-square law)
                        direction = pos_diff / dist
                        repulsive_force = kkappa * direction * (1.0 / dist - 1.0 / d_safe) / (dist ** 2)
                        total_repulse += repulsive_force

                # Apply to this agent's slice of the global state
                f_x[dim] += total_repulse


        # --- 4. Compute barrier-based control
        bb = -(self.kappa * barrier + grad_t + grad_x @ (self.system.A @ x_0 + f_x))

        if bb <= 0:
            u = np.zeros((self.system.B.shape[1],)) + self.B_pinv @ f_x
        else:
            a = grad_x @ self.system.B
            u = (bb / np.linalg.norm(a) ** 2) * a + self.B_pinv @ f_x

        return u



        