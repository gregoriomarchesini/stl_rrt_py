import numpy as np
from   typing import Optional, Union
import cvxpy as cp
from   openmpc.models.linear_system import LinearSystem

from stl_tool.stl.logic import Formula, AndOperator, Node, get_type_and_polytopic_predicate, OrOperator ,UOp, FOp, GOp
from stl_tool.stl.polytope import Polytope
from stl_tool.stl.utils import TimeInterval



class GammaFun:
    def __init__(self):
        

        self.alpha_var   : cp.Variable =  cp.Variable(pos=True)
        self.beta_var    : cp.Variable =  cp.Variable(pos=True)
        self.gamma_0_var : cp.Variable =  cp.Variable(pos=True)
        self.r_var       : cp.Variable =  cp.Variable(pos=True)
    

    @property
    def alpha(self) -> float:
        return self.alpha_var.value
    
    @property
    def beta(self) -> float:
        return self.beta_var.value
    
    @property
    def gamma_0(self) -> float:
        return self.gamma_0_var.value
    
    @property
    def r(self) -> float:
        return self.r_var.value
    
    @alpha.setter
    def alpha(self, value):
        self.alpha_var.value = value
    @beta.setter
    def beta(self, value):
        self.beta_var.value = value
    @gamma_0.setter
    def gamma_0(self, value):
        self.gamma_0_var.value = value
    @r.setter
    def r(self, value):
        self.r_var.value = value


    def _add_(self,polytope: Polytope) :
        return BarrierFunction(polytope = polytope, gamma = self)


class BarrierFunction :
    def __init__(self, polytope: Polytope, gamma: GammaFun):
        
        self._gamma = gamma
        self._polytope = polytope

    @property
    def gamma(self):
        return self._gamma
    
    @property
    def polytope(self):
        return self._polytope

    def E1(self):

        e_vec = [- self.gamma.gamma_0_var/self.gamma.alpha for i in range(self.polytope.num_hyperplanes)]
        e_vec = cp.vstack(e_vec)
        E     = cp.hstack([self.polytope.A, e_vec])
        return E
    
    def E2(self):
        e_vec = [0 for i in range(self.polytope.num_hyperplanes)]
        e_vec = cp.vstack(e_vec)
        E     = cp.hstack([self.polytope.A, e_vec])
        return E
    
    def c1(self) :
        c_vec = cp.vstack([ self.gamma.gamma_0  - self.gamma.r_var for i in range(self.polytope.num_hyperplanes)])
        return c_vec
    
    def c2(self) :
        c_vec = cp.vstack([  - self.gamma.r_var for i in range(self.polytope.num_hyperplanes)])
        return c_vec

    def D(self) :
        return self.polytope.A
    
    def b_vec(self) :
        b_vec = self.polytope.b

    def gamma_var_value(self,t):

        if t <0 :
            raise ValueError("The time must be positive.")
        if t > self.gamma.beta :
            raise ValueError("The time must be less than the beta.")
        
        if t >= self.gamma.alpha  and t <= self.gamma.beta:
            return - self.gamma.r_var 

        if t <= self.gamma.alpha :
            return (self.gamma.gamma_0_var- self.gamma.r_var) + self.gamma.gamma_0_var/self.gamma.alpha * t



def create_barriers_and_time_constraints(formula: Formula) -> tuple[list[BarrierFunction], list[cp.Constraint]] :

    #! create better initial guesses
    root             : Node                     = formula.root
    barriers         : list[BarrierFunction]    = []
    time_constraints : list[cp.Constraint]      = []

    if isinstance(root,OrOperator):
        raise NotImplementedError("OrOperator is not implemented yet. Wait for it. it is coming soon.")
    
    elif isinstance(root,AndOperator):
        formulas = []
        for child in root.children: # saparate each branch into a single formula
            formulas += [Formula(root = child)]
    else:
        formulas = [formula]            

    for formula in formulas:
        try :
            type_formula,polytope = get_type_and_polytopic_predicate(formula)
        except ValueError:
            raise ValueError("At least one of the formulas set in conjunction is not within the currently allowed grammar. Please verify the predicates in the formula.")
        
        root    : Union[FOp,UOp,GOp] = formula.root
        time_interval : TimeInterval = root.interval 

        if type_formula == "G" :

            ## create barrier
            gamma = GammaFun()
            barrier = BarrierFunction(polytope, gamma)
            barriers.append(barrier)
            time_constraints += [gamma.alpha_var == time_interval.a, gamma.beta_var == time_interval.b]

            ## give initial guess
            gamma.alpha_var.value = time_interval.a
            gamma.beta_var.value  = time_interval.b

        elif type_formula == "F" :
            gamma = GammaFun()
            barrier = BarrierFunction(polytope, gamma)
            barriers.append(barrier)
            time_constraints += [gamma.alpha_var >= time_interval.a, 
                                gamma.alpha_var <= gamma.beta_var,
                                gamma.beta_var >= gamma.alpha_var,
                                gamma.beta_var <= time_interval.b]

            ## give initial guess
            gamma.alpha_var.value = time_interval.get_sample()
            gamma.beta_var.value  = gamma.alpha_var.value+0.003

        elif type_formula == "FG":
            time_interval_prime : TimeInterval = root.children[0].interval
            gamma = GammaFun()
            barrier = BarrierFunction(polytope, gamma)
            barriers.append(barrier)
            time_constraints += [gamma.alpha_var >= time_interval.a +  time_interval_prime.a, 
                                gamma.alpha_var <= time_interval.b +  time_interval_prime.a,
                                gamma.beta_var == gamma.alpha_var + (time_interval_prime.b - time_interval_prime.a)]
            
            ## give initial guess
            gamma.alpha_var.value = time_interval.get_sample() + time_interval_prime.a
            gamma.beta_var.value  = gamma.alpha_var.value + (time_interval_prime.b - time_interval_prime.a)

        elif type_formula == "GF":
            time_interval_prime : TimeInterval = root.children[0].interval
            min_repetitions = np.ceil(time_interval.period/time_interval_prime.period)
            
            barriers_rep      : list[BarrierFunction] = []

            gamma_1           = GammaFun()
            barrier_1         = BarrierFunction(polytope, gamma_1)
            time_constraints += [gamma.alpha_var >= time_interval.a +  time_interval_prime.a, 
                                    gamma.alpha_var <= time_interval.a +  time_interval_prime.b,
                                    gamma.beta_var == gamma.alpha_var]
            
            barriers_rep     += [barrier_1]
            ## give initial guess 
            gamma.alpha_var.value = time_interval_prime.get_sample() + time_interval.a
            gamma.beta_var.value  = gamma.alpha_var.value 


            for i in range(1,int(min_repetitions)-1):
                gamma = GammaFun()
                barrier = BarrierFunction(polytope, gamma)
                barriers.append(barrier)
                
                barriers_prev = barriers_rep[i-1]

                time_constraints += [gamma.alpha_var >=  barriers_prev.gamma.alpha_var , 
                                    gamma.alpha_var <=  barriers_prev.gamma.alpha_var + time_interval_prime.period,
                                    gamma.beta_var == gamma.alpha_var]
                
                barriers_rep.append(barrier)
                
                ## give initial guess 
                gamma.alpha_var.value = barriers_prev.gamma.alpha + time_interval_prime.period/2
                gamma.beta_var.value  = gamma.alpha_var.value 


            
            gamma_last = GammaFun()
            barrier_last = BarrierFunction(polytope, gamma_last)
            time_constraints += [gamma.alpha_var >= time_interval.b +  time_interval_prime.a, 
                                gamma.alpha_var <= time_interval.b +  time_interval_prime.b,
                                gamma.beta_var == gamma.alpha_var]
            
            barriers.append(barrier_last)

            ## give initial guess
            gamma.alpha_var.value =  time_interval_prime.get_sample() + time_interval.b
            gamma.beta_var.value  =  gamma.alpha_var.value 

    return barriers, time_constraints




def make_time_schedule(formula: Formula) :

    root             : Node                    = formula.root
    barriers         : list[BarrierFunction]   = []
    time_constraints : list[cp.Constraint]      = []
    
    barriers, time_constraints = create_barriers_and_time_constraints(formula)
    
    # create optimization problem 

    cost = 0
    for barrier_i in barriers:
        for barrier_j in barriers :
            if barrier_i != barrier_j:
                cost += -cp.log(barrier_i.gamma.alpha_var - barrier_j.gamma.alpha_var + 0.2)
                cost += -cp.log(barrier_i.gamma.beta_var - barrier_j.gamma.beta_var   + 0.2)
                cost += -cp.log(barrier_i.gamma.alpha_var - barrier_j.gamma.beta_var  + 0.2)
                cost += -cp.log(barrier_i.gamma.beta_var - barrier_j.gamma.alpha_var  + 0.2)


    problem = cp.Problem(cp.Minimize(cost), time_constraints)
    problem.solve()

    print("Status: ", problem.status)
    print("Optimal value: ", problem.value)



def active_barriers_map(t: float, list_of_barriers: list[BarrierFunction]) -> list[int]:
    """
    Returns the list of active barriers at time t.
    """
    active_barriers = []
    for i, barrier in enumerate(list_of_barriers):
        if barrier.gamma.beta < t:
            active_barriers.append(i)
    return active_barriers



def optimize_barriers(list_of_barriers: list[BarrierFunction], work_space : Polytope, system: LinearSystem):
    
    vertices              = work_space.get_V_representation().T #  dim x num_vertices
    x_dim                 = vertices.shape[0]
    set_of_time_intervals = { barrier.gamma.alpha for barrier in list_of_barriers} | { barrier.gamma.beta for barrier in list_of_barriers} | {0.} # repeated time instants will be counted once in this way
    ordered_sequence      = sorted(set_of_time_intervals)
    k_gain                = 1 # hard coded for now
    constraints           = []

    if system.A_cont is None or system.B_cont is None:
        raise ValueError("The system must posses the continuous time matrices in order to be applied in this framework.")
    

    A_ext = np.block([[system.A_cont,np.zeros((system.A_cont.shape[0],1))],
                      [np.zeros((1,system.A_cont.shape[1])),0]])
    B_ext = np.block([[system.B_cont],
                      [0]])
    P     = np.block([np.zeros(system.A_cont.shape[0]),1])
    

    # dynamic constraints
    for i in range(len(ordered_sequence)-1): # for each interval
        
        s_i = ordered_sequence[i]
        s_i_plus_1 = ordered_sequence[i+1]

        s_j_vec        = np.array([[s_i] for i in range(vertices.shape[1])]) # column vector
        s_j_plus_1_vec = np.array([[s_i_plus_1] for i in range(vertices.shape[1])])
        V_j = np.hstack((np.vstack((vertices,s_j_vec)) , np.vstack((vertices,s_j_plus_1_vec)))) # space time vertices
        U_j = cp.Variable((V_j.shape[0], V_j.shape[1]))                                         # spece of control input vertices

        for active_task_index in active_barriers_map(s_i, list_of_barriers): # for each active map
            barrier = list_of_barriers[active_task_index]
            
            # select correct section
            if s_i <= barrier.gamma.alpha and s_i_plus_1 <= barrier.gamma.alpha : # the (s_i,s_i_plus_1) in [0,\alpha_l]
                E = barrier.E1()
                c = barrier.c1()
            else:
                E = barrier.E2()
                c = barrier.c2()


            for j in range(V_j.shape[1]): # for each vertex
                vertex = V_j[:,j]
                u_vertex = U_j[:,j]
                dyn = A_ext @ vertex + B_ext @ u_vertex + P
                constraints += [E @ dyn >= k_gain * (E @ vertex + c)]


    # Inclusion constraints
    betas = { barrier.gamma.beta for barrier in list_of_barriers} | {0.}
    betas_sorted = sorted(betas)
    zeta_vars = cp.Variable(( x_dim, len(list_of_barriers)))
    for l in range(1,len(betas)):
        beta = betas_sorted[l]
        for l_tilde in active_barriers_map(betas_sorted[i-1], list_of_barriers) :
           
           
           D_matrix  = list_of_barriers[l_tilde].D()
           gamma     = list_of_barriers[l_tilde].gamma_var_value(beta)
           gamma_vec = cp.vstack([gamma for i in range(x_dim)])
           b_vec     = list_of_barriers[l_tilde].b_vec()

           constraints += [D_matrix @ zeta_vars[:,l] - b_vec + gamma_vec >= 0]
        

    # constaints on the parameters 
    for barrier in list_of_barriers :
        constraints += [barrier.gamma.r_var >= 0., barrier.gamma.gamma_0_var >= 0.]








                

            


    
    

    
















