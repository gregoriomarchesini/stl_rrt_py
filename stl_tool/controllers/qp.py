from stl_tool.stl import ContinuousLinearSystem,TimeVaryingConstraint
from stl_tool.polyhedron import Polyhedron
import cvxpy as cp
import numpy as np


class QPController:
    def __init__(self, system : ContinuousLinearSystem, input_bounds : Polyhedron , constraints : list[TimeVaryingConstraint]):

        self.system                   : ContinuousLinearSystem      = system
        self.input_bounds             : Polyhedron                  = input_bounds
        self.time_varying_constraints : list[TimeVaryingConstraint] = constraints
        self.dt                       : float                       = system.dt
        
        self.x_now  = cp.Variable((system.A.shape[0],), name = "x")
        self.t_next = cp.Parameter()

        self.u_var   = cp.Variable((system.B.shape[1],), name = "u")
        self.switch  = cp.Parameter(len(constraints), boolean = True,name = "switch")
        self.problem : cp.Problem = None

        self._set_up()

    
    def _set_up(self) :
        

        cost        = cp.sum_squares(self.u_var)
        constraints = []


        for jj,constraint in enumerate(self.time_varying_constraints):
            H = constraint.H
            b = constraint.b

            x_next = (self.system.A @ self.x_now + self.system.B @ self.u_var)*self.dt + self.x_now

            z_next = cp.hstack([x_next, self.t_next])

            constraints.append(H @ z_next <= b + (1-self.switch[jj])*1E6)

        # input bounds
        constraints += [self.input_bounds.A @ self.u_var <= self.input_bounds.b]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def get_input(self, x_0 : np.ndarray, t_0 : float ):

        t_next = t_0 + self.dt
        
        for jj,self.time_varying_constraints in range(len(self.time_varying_constraints)) :
            if t_next >= self.time_varying_constraints[jj].start_time and t_next <= self.time_varying_constraints[jj].end_time :
                self.switch[jj].value = 1
            else :
                self.switch[jj].value = 0
        
        self.t_next.value = t_next
        self.x_now.value = x_0
        self.problem.solve()
        u = self.u_var.value
        return u



        