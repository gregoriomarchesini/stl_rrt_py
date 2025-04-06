import numpy as np
from   typing import Optional, Union
import cvxpy as cp
from   openmpc import LinearSystem
from matplotlib import pyplot as plt


from stl_tool.stl.logic import Formula, AndOperator, Node, get_type_and_polytopic_predicate, OrOperator ,UOp, FOp, GOp
from stl_tool.polytope import Polytope
from stl_tool.stl.utils import TimeInterval



class GammaFun:
    def __init__(self):
        

        self.alpha_var   : cp.Variable =  cp.Variable(pos=True)
        self.beta_var    : cp.Variable =  cp.Variable(pos=True)
        self.gamma_0_var : cp.Variable =  cp.Variable(pos=True)
        self.r_var       : cp.Variable =  cp.Variable(pos=True)

        self._switch_times_already_resolved : bool = False
    

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

    @property
    def switch_times_already_resolved(self) -> bool:
        return self._switch_times_already_resolved


    def _add_(self,polytope: Polytope) :
        return BarrierFunction(polytope = polytope, gamma = self)


class BarrierFunction :
    def __init__(self, polytope: Polytope, gamma: GammaFun):
        
        self._gamma      = gamma
        self._polytope   = polytope
        self._task_type  = None

    @property
    def gamma(self):
        return self._gamma
    
    @property
    def polytope(self):
        return self._polytope
    
    @property
    def task_type(self):
        return self._task_type
    
    @task_type.setter
    def task_type(self, value):
        if not isinstance(value, str):
            raise ValueError("Task type must be a string.")

    def E1(self):
        
        A = - self.polytope.A  # this is because a polytope is defined as Ax<= b, but forward invariance properties are defined for a polytope in the form Ax>= b
        e_vec = [- self.gamma.gamma_0_var/self.gamma.alpha for i in range(self.polytope.num_hyperplanes)]
        e_vec = cp.vstack(e_vec)
        E     = cp.hstack([A, e_vec])
        return E
    
    def E2(self):
        A = - self.polytope.A  # this is because a polytope is defined as Ax<= b, but forward invariance properties are defined for a polytope in the form Ax>= b
        e_vec = [0 for i in range(self.polytope.num_hyperplanes)]
        e_vec = cp.vstack(e_vec)
        E     = cp.hstack([A , e_vec])
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



def active_barriers_map(t: float, list_of_barriers: list[BarrierFunction]) -> list[int]:
    """
    Returns the list of active barriers at time t.
    """
    active_barriers = []
    for i, barrier in enumerate(list_of_barriers):
        if barrier.gamma.beta > t:
            active_barriers.append(i)
    return active_barriers


class TasksOptimizer:
    
    def __init__(self,formula : Formula, workspace: Polytope) :
        
        self.formula           : Formula                = formula
        self._varphi_formulas  : list[Formula]          = [] # subformulas G,F,FG,GF
        self._barriers         : list[BarrierFunction]  = []
        self._time_constraints : list[cp.Constraint]    = []
        self._workspace        : Polytope               = workspace

        if workspace.is_open:
            raise ValueError("The workspace is an open Polyhedron. Please provide a closed polytope")

    def _create_barriers_and_time_constraints(self) :

        #! create better initial guesses
        barriers         : list[BarrierFunction]    = []
        time_constraints : list[cp.Constraint]      = []

        if isinstance(self.formula.root,OrOperator):
            raise NotImplementedError("OrOperator is not implemented yet. Wait for it. it is coming soon.")
        
        elif isinstance(self.formula.root,AndOperator):
            for child in self.formula.root.children: # saparate each branch into a single formula
                self._varphi_formulas += [Formula(root = child)]
        else:
            self._varphi_formulas = [self.formula]            
        
        for formula in self._varphi_formulas:
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
                barrier.task_type = type_formula
                barriers.append(barrier)
                time_constraints += [gamma.alpha_var == time_interval.a, gamma.beta_var == time_interval.b]

                ## give initial guess
                gamma.alpha_var.value = time_interval.a
                gamma.beta_var.value  = time_interval.b

            elif type_formula == "F" :
                gamma = GammaFun()
                barrier = BarrierFunction(polytope, gamma)
                barrier.task_type = type_formula
                barriers.append(barrier)
                time_constraints += [gamma.alpha_var >= time_interval.a, 
                                    gamma.beta_var   >= gamma.alpha_var,
                                    gamma.beta_var   <= time_interval.b]

                ## give initial guess
                gamma.alpha_var.value = time_interval.get_sample()
                gamma.beta_var.value  = gamma.alpha_var.value+0.003

            elif type_formula == "FG":
                time_interval_prime : TimeInterval = root.children[0].interval
                gamma = GammaFun()
                barrier = BarrierFunction(polytope, gamma)
                barrier.task_type = type_formula
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
                barrier_1.task_type = type_formula
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
                    barrier.task_type = type_formula
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
                barrier_last.task_type = type_formula
                time_constraints += [gamma.alpha_var >= time_interval.b +  time_interval_prime.a, 
                                    gamma.alpha_var <= time_interval.b +  time_interval_prime.b,
                                    gamma.beta_var == gamma.alpha_var]
                
                barriers.append(barrier_last)

                ## give initial guess
                gamma.alpha_var.value =  time_interval_prime.get_sample() + time_interval.b
                gamma.beta_var.value  =  gamma.alpha_var.value 

        for barrier in barriers:  
            print("Barrier alpha: ", barrier.gamma.alpha_var.value)
            print("Barrier beta: ", barrier.gamma.beta_var.value)

        self._barriers         = barriers
        self._time_constraints = time_constraints

    def make_time_schedule(self) :

        
        self._create_barriers_and_time_constraints()
        
        # create optimization problem 

        cost = 0
        normalizer = 50
        for barrier_i in self._barriers:
            cost += cp.exp(- (barrier_i.gamma.beta_var - barrier_i.gamma.alpha_var) - normalizer)
            for barrier_j in self._barriers :
                if barrier_i != barrier_j and barrier_i.task_type != "G":
                    cost += cp.exp(-(barrier_i.gamma.alpha_var - barrier_j.gamma.alpha_var)- normalizer )
                    cost += cp.exp(-(barrier_i.gamma.alpha_var - barrier_j.gamma.beta_var )- normalizer )
                    
                    cost += cp.exp(-(barrier_i.gamma.beta_var - barrier_j.gamma.beta_var )- normalizer)
                    cost += cp.exp(-(barrier_i.gamma.beta_var - barrier_j.gamma.alpha_var)- normalizer)


        problem = cp.Problem(cp.Minimize(cost), self._time_constraints)
        problem.solve(warm_start=True, verbose=True)

        print("Status: ", problem.status)
        print("Optimal value: ", problem.value)

        for barrier in self._barriers:  
            print("Barrier alpha: ", barrier.gamma.alpha_var.value)
            print("Barrier beta: ", barrier.gamma.beta_var.value)

    def plot_time_schedule(self) -> None:
        
        tasks :list[dict] = []
        for formula in self._varphi_formulas:
            try :
                type_formula , polytope = get_type_and_polytopic_predicate(formula)
            except ValueError:
                raise ValueError("At least one of the formulas set in conjunction is not within the currently allowed grammar. Please verify the predicates in the formula.")

            # Sample task data: (start_time, duration)
            
            if type_formula == "G" :
                # Plotting the G operator
                start_time = formula.root.interval.a
                duration   = formula.root.interval.b - start_time
                tasks.append({'start_time': start_time, 'duration': duration})
            elif type_formula == "F" :
                # Plotting the F operator
                start_time = formula.root.interval.a
                duration   = formula.root.interval.b - start_time
                tasks.append({'start_time': start_time, 'duration': duration})

            elif type_formula == "FG" :
                # Plotting the FG operator
                start_time = formula.root.interval.a + formula.root.children[0].interval.a
                duration   =  (formula.root.interval.b + formula.root.children[0].interval.b) - start_time
                tasks.append({'start_time': start_time, 'duration': duration})

            elif type_formula == "GF" :
                # Plotting the GF operator
                start_time = formula.root.interval.a + formula.root.children[0].interval.a
                duration   =  (formula.root.interval.b + formula.root.children[0].interval.b) - start_time
                tasks.append({'start_time': start_time, 'duration': duration})
            else:
                raise ValueError("Unsupported formula type for plotting.")
               
        print("Tasks: ", tasks)
        print(self._varphi_formulas)
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot each task as a thin bar on its own y-row
        for i, task in enumerate(tasks):
            ax.broken_barh([(task["start_time"], task["duration"])], (i - 0.4, 0.8), facecolors='tab:blue')

        # Labeling
        ax.set_xlabel('Time')
        ax.set_ylabel('Task number')
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([f'Task {i+1}' for i in range(len(tasks))])
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def optimize_barriers(self, input_bounds: Polytope ,system : LinearSystem, x_0 : np.ndarray) -> None:
        
        
        if input_bounds.is_open:
            raise ValueError("The input bounds are an open Polyhedron. Please provide a closed polytope")
        
        if not isinstance(system, LinearSystem):
            raise ValueError("The system must be a LinearSystem.")
        
        x_0 = x_0.flatten()
        if len(x_0) != self._workspace.num_dimensions:
            raise ValueError("The initial state must be a vector of the same dimension as the workspace.")

        
        vertices              = self._workspace.vertices # dim x num_vertices
        x_dim                 = self._workspace.num_dimensions
        set_of_time_intervals = { barrier.gamma.alpha for barrier in self._barriers} | { barrier.gamma.beta for barrier in self._barriers} | {0.} # repeated time instants will be counted once in this way
        ordered_sequence      = sorted(set_of_time_intervals)
        k_gain                = 1 # hard coded for now
        constraints           = []

        if system.A_cont is None or system.B_cont is None:
            raise ValueError("The system must posses the continuous time matrices in order to be applied in this framework. Make sure you system derives from a continuous time system by calling the method c2c of the class Linear System")
        
        # create extended dynamics
        A_ext = np.block([[system.A_cont,np.zeros((system.A_cont.shape[0],0))],
                        [np.zeros((1,system.A_cont.shape[1]))          ,0]])
        B_ext = np.block([[system.B_cont],
                        [0]])
        P     = np.block([np.zeros(system.A_cont.shape[0]),1])
        
        # dynamic constraints
        for i in range(len(ordered_sequence)-1): # for each interval
            
            # Get two consecutive time intervals in the sequence.
            s_i        = ordered_sequence[i]
            s_i_plus_1 = ordered_sequence[i+1]



            # Create vertices set. 
            s_j_vec        = np.array([[s_i] for i in range(vertices.shape[1])]) # column vector
            s_j_plus_1_vec = np.array([[s_i_plus_1] for i in range(vertices.shape[1])])
            V_j            = np.hstack((np.vstack((vertices,s_j_vec)) , np.vstack((vertices,s_j_plus_1_vec)))) # space time vertices
            U_j            = cp.Variable((V_j.shape[0], V_j.shape[1]))                                         # spece of control input vertices
             

            # Forward-invariance constraints. 
            for active_task_index in active_barriers_map(s_i, self._barriers): # for each active map
                barrier = self._barriers[active_task_index]
                
                # select correct section
                if s_i <= barrier.gamma.alpha and s_i_plus_1 <= barrier.gamma.alpha : # then (s_i,s_i_plus_1) in [0,\alpha_l]
                    E = barrier.E1()
                    c = barrier.c1()
                else: # then (s_i,s_i_plus_1) in [\alpha_l,\beta_l]
                    E = barrier.E2()
                    c = barrier.c2()

                for j in range(V_j.shape[1]): # for each vertex
                    vertex       = V_j[:,j]
                    u_vertex     = U_j[:,j]
                    dyn          = A_ext @ vertex + B_ext @ u_vertex + P
                    constraints += [E @ dyn >= k_gain * (E @ vertex + c)]


        # Inclusion constraints
        betas = { barrier.gamma.beta for barrier in self._barriers} | {0.}
        betas_sorted = sorted(betas)
        zeta_vars = cp.Variable(( x_dim, len(self._barriers)))
        
        for l in range(1,len(betas)):
            beta = betas_sorted[l]
            
            for l_tilde in active_barriers_map(betas_sorted[i-1], self._barriers) : # barriers active at lim t-> - beta_l is equal to the one active at time beta_{l-1}
            
                D_matrix  = self._barriers[l_tilde].D()
                gamma     = self._barriers[l_tilde].gamma_var_value(beta)
                gamma_vec = cp.vstack([gamma for i in range(x_dim)])
                b_vec     = self._barriers[l_tilde].b_vec()

                constraints += [D_matrix @ zeta_vars[:,l] - b_vec + gamma_vec >= 0]
            

        # constaints on the parameters 
        for barrier in self._barriers :
            constraints += [barrier.gamma.r_var >= 0., barrier.gamma.gamma_0_var >= 0.]








                

            


    
    

    
















