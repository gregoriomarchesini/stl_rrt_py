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
        

        self.alpha_var   : cp.Variable =  cp.Variable(nonneg=True)
        self.beta_var    : cp.Variable =  cp.Variable(nonneg=True)
        self.gamma_0_var : cp.Variable =  cp.Variable(nonneg=True)
        self.r_var       : cp.Variable =  cp.Variable(pos=True)

    @property
    def alpha(self) -> float:
        if self.alpha_var.value is None:
            raise ValueError("Alpha variable has not been set yet.")
        return self.alpha_var.value
    
    @property
    def beta(self) -> float:
        if self.beta_var.value is None:
            raise ValueError("Beta variable has not been set yet.")
        return self.beta_var.value
    
    @property
    def gamma_0(self) -> float:
        if self.gamma_0_var.value is None:
            raise ValueError("Gamma_0 variable has not been set yet.")
        return self.gamma_0_var.value
    
    @property
    def r(self) -> float:
        if self.r_var.value is None:
            raise ValueError("R variable has not been set yet.")
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

    def upsilon(self,t:float)-> int :
        t = float(t)

        if t >= 0. and t < self.alpha:
            return 1
        elif t >= self.alpha and t <= self.beta:
            return 2
        else :
            raise ValueError("The given time time is outside the range [0,beta].")
            
        


    def _add_(self,polytope: Polytope) :
        return BarrierFunction(polytope = polytope, gamma = self)

class BarrierFunction :
    """
    Note that polytope representaton in cddlib is given as Ax<= b
    But in the paper we have Dx + c >= 0 and thus -Dx<= c.

    So In all the equations we get D = -A and c= b
    
    """
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

    
    def D(self):
        D = - self.polytope.A
        return D
    
    def e1(self):
        
        e_vec = - self.gamma.gamma_0_var/self.gamma.alpha *np.ones(self.polytope.num_hyperplanes) 
        e_vec = cp.hstack(e_vec)
        return e_vec
    
    def e2(self):
        e_vec = np.zeros(self.polytope.num_hyperplanes)
        e_vec = cp.hstack(e_vec)
        return e_vec
    
    def g1(self) :
        g_vec = (self.gamma.gamma_0_var  - self.gamma.r_var)*np.ones(self.polytope.num_hyperplanes) 
        return g_vec
    
    def g2(self) :
        g_vec = -self.gamma.r_var *np.ones(self.polytope.num_hyperplanes) 
        return g_vec

    def c(self) :
        return self.polytope.b
    
    def gamma_var_value(self,t):

        if t <0 :
            raise ValueError("The time must be positive.")
        if t > self.gamma.beta :
            raise ValueError("The time must be less than the beta.")
        
        if t >= self.gamma.alpha  and t <= self.gamma.beta:
            return - self.gamma.r_var 

        if t <= self.gamma.alpha :
            return (self.gamma.gamma_0_var- self.gamma.r_var) + self.gamma.gamma_0_var/self.gamma.alpha * t



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

        #! create better initial time guesses
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
        problem.solve(warm_start=True, verbose=False)

        print("Status: ", problem.status)
        print("Optimal value: ", problem.value)

        for barrier in self._barriers:  
            print("Barrier alpha: ", barrier.gamma.alpha_var.value)
            print("Barrier beta: ", barrier.gamma.beta_var.value)

    
    def active_barriers_map(self,t: float) -> list[int]:
        """
        Returns the list of active barriers at time t.
        """
        active_barriers = []
        for i, barrier in enumerate(self._barriers):
            if barrier.gamma.beta > t:
                active_barriers.append(i)
        return active_barriers
    
    
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


        max_beta = 0 
        for barrier in self._barriers:
            max_beta = max(max_beta,barrier.gamma.beta)
        
        # plot the cardinality of the active set map
        fig, ax = plt.subplots(figsize=(10, 4))
        t = np.linspace(0, max_beta, int(max_beta*100))
        card = []
        for time in t:
            card += [len(self.active_barriers_map(time))]
        
        ax.plot(t, card, label='Cardinality of Active Set')
        
        plt.tight_layout()
        plt.show()


    
    
    def optimize_barriers(self, input_bounds: Polytope ,system : LinearSystem, x_0 : np.ndarray):
        
        
        if input_bounds.is_open:
            raise ValueError("The input bounds are an open Polyhedron. Please provide a closed polytope")
        
        if not isinstance(system, LinearSystem):
            raise ValueError("The system must be a LinearSystem.")
        
        x_0 = x_0.flatten()
        if len(x_0) != self._workspace.num_dimensions:
            raise ValueError("The initial state must be a vector of the same dimension as the workspace.")

        
        vertices              = self._workspace.vertices.T # matrix [dim x,num_vertices]
        x_dim                 = self._workspace.num_dimensions
        set_of_time_intervals = list({ barrier.gamma.alpha for barrier in self._barriers} | { barrier.gamma.beta for barrier in self._barriers} | {0.}) # repeated time instants will be counted once in this way
        ordered_sequence      = sorted(set_of_time_intervals)
        k_gain                = 1 # hard coded for now
        constraints  :list[cp.Constraint]  = []

        if system.A_cont is None or system.B_cont is None:
            raise ValueError("The system must posses the continuous time matrices in order to be applied in this framework. Make sure you system derives from a continuous time system by calling the method c2c of the class Linear System")
        
        A = system.A_cont
        B = system.B_cont

        
        # dynamic constraints
        for jj in range(len(ordered_sequence)-1): # for each interval
            
            # Get two consecutive time intervals in the sequence.
            s_j        = ordered_sequence[jj]
            s_j_plus_1 = ordered_sequence[jj+1]

            # Create vertices set. 
            s_j_vec        = np.array([s_j for i in range(vertices.shape[1])]) # column vector
            s_j_plus_1_vec = np.array([s_j_plus_1 for i in range(vertices.shape[1])])
            V_j            = np.hstack((np.vstack((vertices,s_j_vec)) , np.vstack((vertices,s_j_plus_1_vec)))) # space time vertices
            U_j            = cp.Variable((system.size_input, V_j.shape[1]))                                         # spece of control input vertices
            
            # Input constraints.
            for kk in range(U_j.shape[1]):
                u_kk = U_j[:,kk]
                constraints += [input_bounds.A @ u_kk <= input_bounds.b]

            # Forward-invariance constraints. 
            for active_task_index in self.active_barriers_map(s_j): # for each active task
                barrier = self._barriers[active_task_index]
                
                # select correct section (equivalent to upsilon)
                if barrier.gamma.upsilon(s_j) == 1 : # then (s_i,s_i_plus_1) in [0,\alpha_l]
                    e = barrier.e1()
                    g = barrier.g1()
                else: # then (s_i,s_i_plus_1) in [\alpha_l,\beta_l]
                    e = barrier.e2()
                    g = barrier.g2()

                c = barrier.c()
                D = barrier.D()

                for kk in range(V_j.shape[1]): # for each vertex
                    eta_kk   = V_j[:,kk]
                    u_kk     = U_j[:,kk]

                    eta_kk_not_time     = eta_kk[:-1]
                    time                = eta_kk[-1]

                    dyn                 = A @ eta_kk_not_time + B @ u_kk 
                    constraints        += [D @ dyn + e >= -k_gain * (D @ eta_kk_not_time + e*time + c + g)]

        
        # Inclusion constraints
        betas        = list({ barrier.gamma.beta for barrier in self._barriers} | {0.}) # set inside the list removes duplicates if any.
        betas        = sorted(betas)
        zeta_vars    = cp.Variable(( x_dim, len(betas)))

        # Impose zeta vars in the workspace
        for kk in range(zeta_vars.shape[1]):
            zeta_kk       =  zeta_vars[:,kk]
            constraints  += [self._workspace.A @ zeta_kk <= self._workspace.b]
            
        for l in range(1,len(betas)):
            beta_l = betas[l]
            zeta_l = zeta_vars[:,l]

            for l_tilde in self.active_barriers_map(betas[l-1]) : # barriers active at lim t-> - beta_l is equal to the one active at time beta_{l-1}
                
                print("Value of upsilon")
                print(self._barriers[l_tilde].gamma.upsilon(beta_l))
                if self._barriers[l_tilde].gamma.upsilon(beta_l) == 1: # checking for the value of upsilon
                    e = self._barriers[l_tilde].e1()
                    g = self._barriers[l_tilde].g1()
                else:
                    e = self._barriers[l_tilde].e2()
                    g = self._barriers[l_tilde].g2()

                D = self._barriers[l_tilde].D()  
                c = self._barriers[l_tilde].c() 
                
                constraints += [D @ zeta_l + e * beta_l + c + g >= 0]
        
        # set the zeta at beta=0 zero and conclude
        # initial state constraint
        
        zeta_0 = zeta_vars[:,0]
        for barrier in self._barriers:
            # at time beta=0 all tasks are active and they are in the first linear section of gamma
            e = barrier.e1()
            g = barrier.g1()
            
            c = barrier.c()
            D = barrier.D()
            constraints += [D @ zeta_0 + e*0 + c + g >= 0]

        constraints += [zeta_0 == x_0]
  

        # create problem and solve it
        cost = 0
        for barrier in self._barriers:
            cost += -barrier.gamma.r_var

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        print("===========================================================")
        print("Summary")
        print("===========================================================")
        print("Status        : ", problem.status)
        print("Optimal value : ", problem.value)
        print("Obtained parameters :")

        for barrier in self._barriers:
            print("Operator        : ", barrier.task_type)
            print("Barrier alpha   : ", barrier.gamma.alpha_var.value)
            print("Barrier beta    : ", barrier.gamma.beta_var.value)
            print("Barrier gamma_0 : ", barrier.gamma.gamma_0_var.value)
            print("Barrier r       : ", barrier.gamma.r_var.value)

        return problem.status


                

            


    
    

    
















