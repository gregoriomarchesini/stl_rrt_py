import numpy as np
from   typing import Optional, Union
import cvxpy as cp
from   openmpc import LinearSystem
from matplotlib import pyplot as plt
import json


from stl_tool.stl.logic import Formula, AndOperator, Node, get_type_and_polytopic_predicate, OrOperator ,UOp, FOp, GOp
from stl_tool.polytope  import Polytope
from stl_tool.stl.utils import TimeInterval


class BarrierFunction :
    """
    Note that polytope representaton in cddlib is given as Ax<= b
    But in the paper we have Dx + c >= 0 and thus -Dx<= c.

    So In all the equations we get D = -A and c= b
    
    """
    def __init__(self, polytope: Polytope):
        
        self._polytope   = polytope
        self._task_type  = None

        self.alpha_var   : cp.Variable =  cp.Variable(nonneg=True)
        self.beta_var    : cp.Variable =  cp.Variable(nonneg=True)
        self.gamma_0_var : cp.Variable =  cp.Variable((self._polytope.num_hyperplanes),nonneg=True)
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

        if (t >= 0.) and (t < self.alpha):
            return 1
        elif (t >= self.alpha) and (t <= self.beta):
            return 2
        else :
            raise ValueError("The given time time is outside the range [0,beta].")

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
        if value not in ["G", "F", "FG", "GF"]:
            raise ValueError("Task type must be one of 'G', 'F', 'FG', or 'GF'.")
        self._task_type = value

    
    def D(self):
        D = - self.polytope.A
        return D
    
    def e1_var(self):
        e_vec = - (self.gamma_0_var/self.alpha) 
        return e_vec
    
    def e2_var(self):
        e_vec = np.zeros(self.polytope.num_hyperplanes)
        return e_vec
    
    def g1_var(self) :
        g_vec = (self.gamma_0_var  - self.r_var)
        return g_vec
    
    def g2_var(self) :
        g_vec = -self.r_var *np.ones(self.polytope.num_hyperplanes) 
        return g_vec

    def c(self) :
        return self.polytope.b
    
    def e1_value(self):
        return - self.gamma_0/self.alpha 
    def e2_value(self):
        return np.zeros(self.polytope.num_hyperplanes)
    def g1_value(self):
        return (self.gamma_0 - self.r) 
    def g2_value(self):
        return - self.r * np.ones(self.polytope.num_hyperplanes)
    

    def gamma_var_value(self,t):

        if t <0 :
            raise ValueError("The time must be positive.")
        if t > self.beta :
            raise ValueError("The time must be less than the beta.")
        
        if t >= self.alpha  and t <= self.beta:
            return - self.r_var 

        if t <= self.alpha :
            return (self.gamma_0_var- self.r_var) + self.gamma_0_var/self.alpha * t



class TasksOptimizer:
    
    def __init__(self,formula : Formula, workspace: Polytope) :
        
        self.formula           : Formula                = formula
        self._workspace        : Polytope               = workspace

        self._varphi_formulas  : list[Formula]          = [] # subformulas G,F,FG,GF
        self._barriers         : list[BarrierFunction]  = []
        self._time_constraints : list[cp.Constraint]    = []

        self._time_varying_polytope_constraints : list[TimeVaryingConstraint] = []

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
        

        self.tasks : list[dict] = []
        
        print("============================================================")
        print("Enumerating tasks")
        print("============================================================")
        for formula in self._varphi_formulas:
            try :
                type_formula,polytope = get_type_and_polytopic_predicate(formula)
            except ValueError:
                raise ValueError("At least one of the formulas set in conjunction is not within the currently allowed grammar. Please verify the predicates in the formula.")
            
            root    : Union[FOp,UOp,GOp] = formula.root
            time_interval : TimeInterval = root.interval 

            if type_formula == "G" :

                ## create barrier
                barrier = BarrierFunction(polytope)
                barrier.task_type = type_formula
                barriers.append(barrier)
                time_constraints += [barrier.alpha_var == time_interval.a, barrier.beta_var == time_interval.b]

                ## give initial guess
                barrier.alpha_var.value = time_interval.a
                barrier.beta_var.value  = time_interval.b

                start_time = formula.root.interval.a
                duration   =  formula.root.interval.b - start_time
                self.tasks.append({'start_time': start_time, 'duration': duration, 'type': type_formula})

            elif type_formula == "F" :
                barrier = BarrierFunction(polytope)
                barrier.task_type = type_formula
                barriers.append(barrier)
                time_constraints += [barrier.alpha_var >= time_interval.a, 
                                    barrier.beta_var   == barrier.alpha_var,
                                    barrier.beta_var   <= time_interval.b]

                ## give initial guess
                barrier.alpha_var.value = time_interval.get_sample()
                barrier.beta_var.value  = barrier.alpha_var.value+0.003

                start_time = formula.root.interval.a
                duration   =  formula.root.interval.b - start_time
                self.tasks.append({'start_time': start_time, 'duration': duration, 'type': type_formula})

            elif type_formula == "FG":
                time_interval_prime : TimeInterval = root.children[0].interval
                barrier = BarrierFunction(polytope)
                barrier.task_type = type_formula
                barriers.append(barrier)
                time_constraints += [barrier.alpha_var >= time_interval.a +  time_interval_prime.a, 
                                    barrier.alpha_var <= time_interval.b +  time_interval_prime.a,
                                    barrier.beta_var == barrier.alpha_var + (time_interval_prime.b - time_interval_prime.a)]
                
                ## give initial guess
                barrier.alpha_var.value = time_interval.get_sample() + time_interval_prime.a
                barrier.beta_var.value  = barrier.alpha_var.value + (time_interval_prime.b - time_interval_prime.a)

                start_time = formula.root.interval.a + formula.root.children[0].interval.a
                duration   =  (formula.root.interval.b + formula.root.children[0].interval.b) - start_time
                self.tasks.append({'start_time': start_time, 'duration': duration, 'type': type_formula})

            elif type_formula == "GF":

                start_time = formula.root.interval.a + formula.root.children[0].interval.a
                duration   =  (formula.root.interval.b + formula.root.children[0].interval.b) - start_time
                self.tasks.append({'start_time': start_time, 'duration': duration, 'type': type_formula})


                time_interval_prime : TimeInterval = root.children[0].interval
                min_repetitions = np.ceil(time_interval.period/time_interval_prime.period)
                
                barriers_rep      : list[BarrierFunction] = []

                barrier_1         = BarrierFunction(polytope)
                barrier_1.task_type = type_formula
                time_constraints += [barrier_1 .alpha_var >= time_interval.a +  time_interval_prime.a, 
                                        barrier_1 .alpha_var <= time_interval.a +  time_interval_prime.b,
                                       barrier_1.beta_var == barrier_1.alpha_var]
                
                barriers_rep     += [barrier_1]
                ## give initial guess 
                barrier_1.alpha_var.value = time_interval_prime.get_sample() + time_interval.a
                barrier_1.beta_var.value  = barrier_1.alpha_var.value 


                for i in range(1,int(min_repetitions)-1):
                    barrier = BarrierFunction(polytope)
                    barrier.task_type = type_formula
                    barriers.append(barrier)
                    
                    barriers_prev = barriers_rep[i-1]

                    time_constraints += [barrier.alpha_var >=  barriers_prev.alpha_var , 
                                         barrier.alpha_var <=  barriers_prev.alpha_var + time_interval_prime.period,
                                         barrier.beta_var == barrier.alpha_var]
                     
                    barriers_rep.append(barrier)
                    
                    ## give initial guess 
                    barrier.alpha_var.value = barriers_prev.alpha + time_interval_prime.period/2
                    barrier.beta_var.value  = barrier.alpha_var.value 


                
                barrier_last = BarrierFunction(polytope)
                barrier_last.task_type = type_formula
                time_constraints += [barrier_last.alpha_var >= time_interval.b +  time_interval_prime.a, 
                                     barrier_last.alpha_var <= time_interval.b +  time_interval_prime.b,
                                     barrier_last.beta_var == barrier_last.alpha_var]
                
                barriers.append(barrier_last)

                ## give initial guess
                barrier_last.alpha_var.value =  time_interval_prime.get_sample() + time_interval.b
                barrier_last.beta_var.value  =  barrier_last.alpha_var.value 
                
            
            print("Found Tasks of type: ", type_formula)
            print("Start time: ", start_time)
            print("Duration: ", duration)
            print("---------------------------------------------")
        print("====== Enumeration completed =======================")

        self._barriers         = barriers
        self._time_constraints = time_constraints

    def _make_high_order_corrections(self, system : LinearSystem, k_gain : float ):

        """
        High order barrier funcitons are required if a given predicate is not directly controllable.

        Take for example a linear system with dynamics x_dot = Ax = Bu and take a predicate of the form  a x + c >= 0.
        Then it can be that c^TB = 0 such that the barrier constraint cT (Ax + Bu) \geq k_gain*(a x + c ) can only be satisfied if the system satisfies it, but there is no 
        controllability on the system. We should then consider the higher order derivatives of the predicate.
        
        """

        if not system.is_controllable():
            raise ValueError("The provided system is not controllable. As of now, only controllable systems can be considered for barriers optimization.")
        

       

        A = system.A_cont
        B = system.B_cont
        I = np.eye(system.size_state)

        print("================================================")
        print("Correcting barriers for high order systems")
        print("================================================")
        for barrier in self._barriers :
            D = barrier.D()
            c = barrier.c()
            still_not_right_order  = True
            order = 0
            while still_not_right_order : #(for controllable systems this will stop after at most a number of iterations equal to the state space)
                D = D@(A + I*k_gain)
                # if there is at least one row of all zeros then you must go on
                for jj in range(D.shape[0]):
                    if np.all(D[jj,:] == 0):
                        pass
                    else:
                        # controllability is satisfied
                        still_not_right_order = False 
                        break
                order += 1
            
            print("Found barrier function of order : ", order)
            new_polytope = Polytope(A =  -D  , b = k_gain**order * c) # just change of sign needed for the descrption of the politope (Barrier is Dx + c >= but polytope is Ax <= b)
            barrier._polytope = new_polytope
        

    def make_time_schedule(self) :

        self._create_barriers_and_time_constraints()
        
        # create optimization problem 

        cost = 0
        normalizer = 50.
        lift_up_factor = 10.
        for barrier_i in self._barriers:
            cost += cp.exp(- (barrier_i.beta_var - barrier_i.alpha_var) - normalizer)
            for barrier_j in self._barriers :
                if barrier_i != barrier_j and barrier_i.task_type != "G":
                    cost += lift_up_factor*cp.exp(-(barrier_i.alpha_var - barrier_j.alpha_var)/normalizer )
                    cost += lift_up_factor*cp.exp(-(barrier_i.alpha_var - barrier_j.beta_var )/normalizer )
                    
                    cost += lift_up_factor*cp.exp(-(barrier_i.beta_var - barrier_j.beta_var )/normalizer)
                    cost += lift_up_factor*cp.exp(-(barrier_i.beta_var - barrier_j.alpha_var)/normalizer)


        problem = cp.Problem(cp.Minimize(cost), self._time_constraints)
        problem.solve(warm_start=True, verbose=False)
        print("===========================================================")
        print("Times Schedule completed")
        print("===========================================================")
        print("Number of barrier functions created: ", len(self._barriers))
        print("Times optimization status: ", problem.status)
        
        print("Listing alpha and beta values per task :")
        for barrier in self._barriers:  
            print("Operator      : ", barrier.task_type)
            print("Barrier alpha : ", barrier.alpha_var.value)
            print("Barrier beta  : ", barrier.beta_var.value)
            print("-----------------------------------------------------")

    
    def active_barriers_map(self,t: float) -> list[int]:
        """
        Returns the list of active barriers at time t.
        """
        active_barriers = []
        for i, barrier in enumerate(self._barriers):
            if barrier.beta > t:
                active_barriers.append(i)
        return active_barriers
    
    
    def plot_time_schedule(self) -> None:
        
        
               
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot each task as a thin bar on its own y-row
        for i, task in enumerate(self.tasks):
            ax.broken_barh([(task["start_time"], task["duration"])], (i - 0.4, 0.8), facecolors='tab:blue')

        # Labeling
        ax.set_xlabel('Time')
        ax.set_ylabel('Task number')
        ax.set_yticks(range(len(self.tasks)))
        ax.set_yticklabels([f'Task {i+1}' for i in range(len(self.tasks))])
        ax.grid(True)


        max_beta = 0.
        for barrier in self._barriers:
            max_beta = max(max_beta,barrier.beta)
        
        # plot the cardinality of the active set map
        fig, ax = plt.subplots(figsize=(10, 4))
        t = np.linspace(0, max_beta, int(max_beta*100))
        card = []
        for time in t:
            card += [len(self.active_barriers_map(time))]
        
        ax.plot(t, card, label='Cardinality of Active Set')
        
        plt.tight_layout()
        plt.show()


    
    
    def optimize_barriers(self, input_bounds: Polytope ,system : LinearSystem, x_0 : np.ndarray, gain: float = 1.0) :
        
        
        if input_bounds.is_open:
            raise ValueError("The input bounds are an open Polyhedron. Please provide a closed polytope")
        
        if not isinstance(system, LinearSystem):
            raise ValueError("The system must be a LinearSystem.")
        
        x_0 = x_0.flatten()
        if len(x_0) != self._workspace.num_dimensions:
            raise ValueError("The initial state must be a vector of the same dimension as the workspace.")

        
        vertices              = self._workspace.vertices.T # matrix [dim x,num_vertices]
        x_dim                 = self._workspace.num_dimensions
        set_of_time_intervals = list({ barrier.alpha for barrier in self._barriers} | { barrier.beta for barrier in self._barriers} | {0.}) # repeated time instants will be counted once in this way
        ordered_sequence      = sorted(set_of_time_intervals)
        k_gain                = cp.Parameter(nonneg=True) #! (when program fails try to increase control input and increase the k_gain. Carefully analyze the situation. Usually it should be at leat equal to 1.)
        k_gain.value          = gain
        
        self._make_high_order_corrections(system = system, k_gain = k_gain.value)
        

        if system.A_cont is None or system.B_cont is None:
            raise ValueError("The system must posses the continuous time matrices in order to be applied in this framework. Make sure you system derives from a continuous time system by calling the method c2c of the class Linear System")
        
        A = system.A_cont
        B = system.B_cont

        constraints  :list[cp.Constraint]  = []

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
                if barrier.upsilon(s_j) == 1 : # then (s_i,s_i_plus_1) in [0,\alpha_l]
                    e = barrier.e1_var()
                    g = barrier.g1_var()
                else: # then (s_i,s_i_plus_1) in [\alpha_l,\beta_l]
                    e = barrier.e2_var()
                    g = barrier.g2_var()

                c = barrier.c()
                D = barrier.D()
                
                for ii in range(V_j.shape[1]): # for each vertex
                    eta_ii   = V_j[:,ii]
                    u_ii     = U_j[:,ii]

                    eta_ii_not_time     = eta_ii[:-1]
                    time                = eta_ii[-1]
                    dyn                 = (A @ eta_ii_not_time + B @ u_ii)
                    constraints        += [D @ dyn + e + k_gain * (D @ eta_ii_not_time + e*time + c + g) >= 0]

        
        # Inclusion constraints
        betas        = list({ barrier.beta for barrier in self._barriers} | {0.}) # set inside the list removes duplicates if any.
        betas        = sorted(betas)
        zeta_vars    = cp.Variable(( x_dim, len(betas)))

        # Impose zeta vars in the workspace
        for kk in range(1,zeta_vars.shape[1]):
            zeta_kk       =  zeta_vars[:,kk]
            constraints  += [self._workspace.A @ zeta_kk <= self._workspace.b]
            
        for l in range(1,len(betas)):
            beta_l = betas[l]
            zeta_l = zeta_vars[:,l]

            for l_tilde in self.active_barriers_map(betas[l-1]) : # barriers active at lim t-> - beta_l is equal to the one active at time beta_{l-1}
                
                if self._barriers[l_tilde].upsilon(beta_l) == 1: # checking for the value of upsilon
                    e = self._barriers[l_tilde].e1_var()
                    g = self._barriers[l_tilde].g1_var()
                else:
                    e = self._barriers[l_tilde].e2_var()
                    g = self._barriers[l_tilde].g2_var()

                D = self._barriers[l_tilde].D()  
                c = self._barriers[l_tilde].c() 
                constraints += [D @ zeta_l + e * beta_l + c + g >= 0]
        
        # set the zeta at beta=0 zero and conclude
        # initial state constraint
        zeta_0  = zeta_vars[:,0]
        epsilon = 1E-1 # just to be strictly inside
        for barrier in self._barriers:
            # at time beta=0 all tasks are active and they are in the first linear section of gamma
            e = barrier.e1_var()
            g = barrier.g1_var()
            
            c = barrier.c()
            D = barrier.D()

            constraints += [D @ zeta_0 + c + g >= epsilon]
        
        # initial state constraint
        constraints += [zeta_0 == x_0]




        # create problem and solve it
        cost = 0
        for barrier in self._barriers:
            cost += -barrier.r_var
            
        fig,ax = plt.subplots(figsize=(10, 4))    
        problem = cp.Problem(cp.Minimize(cost), constraints)
        good_k_found = False
        for k_val in np.arange(0.01,100,0.1):
            k_gain.value = k_val
            problem.solve(warm_start=True)
            if problem.status == cp.OPTIMAL :
                good_k_found = True
                break
        
        if not good_k_found:
            raise ValueError("A suitable k (the alpha functions for the barriers) was not found. Try to increase the input and retry.")
        
            

        plt.show()



        print("===========================================================")
        print("Barrier functions optimization result")
        print("===========================================================")
        print("Status        : ", problem.status)
        print("Optimal value : ", problem.value)
        print("-----------------------------------------------------------")
        print("Listing parameters per task")

        for barrier in self._barriers:
            print("Operator        : ", barrier.task_type)
            print("Barrier alpha   : ", barrier.alpha_var.value)
            print("Barrier beta    : ", barrier.beta_var.value)
            print("Barrier gamma_0 : ", barrier.gamma_0_var.value)
            print("Barrier r       : ", barrier.r_var.value)
            print("---------------------------------------------------")

        if problem.status != cp.OPTIMAL:
            print("Problem is not optimal. Terminate!")
            exit()
        
        # saving constraints as time_state constraints
        for barrier in self._barriers:
            D = barrier.D()
            c = barrier.c()
            e1 = barrier.e1_value()
            e2 = barrier.e2_value()
            g1 = barrier.g1_value()
            g2 = barrier.g2_value()

            alpha = barrier.alpha
            beta  = barrier.beta
            
            H1 = np.hstack((D,e1[:,np.newaxis]))
            H2 = np.hstack((D,e2[:,np.newaxis]))

            b1 =  c + g1
            b2 =  c + g2

            # convert from the form Hx + c >= 0 to Hx <= b
            H1 = -H1
            H2 = -H2

            self._time_varying_polytope_constraints += [TimeVaryingConstraint(start_time=0., end_time=alpha, H=H1, b=b1), TimeVaryingConstraint(start_time=alpha, end_time=beta, H=H2, b=b2)]
            
             
        return problem.solver_stats
    
    

    def show_time_varying_level_set(self) :
        
        fig, ax = plt.subplots()
        for t in np.linspace(0., self.formula.max_horizon(), 30):
            
            polytopes = []
            for constrain in self._time_varying_polytope_constraints:
                if t <= constrain.end_time and t >= constrain.start_time:
                    H = constrain.H[:,:-1]
                    b = constrain.b
                    e = constrain.H[:,-1]

                    polytope = Polytope(H, b - e*t)
                    polytopes.append(polytope)


            # create intersections
            if len(polytopes) > 0:
                intersection = polytopes[0]
                for i in range(1, len(polytopes)):
                    intersection = intersection.intersect(polytopes[i])
                
                intersection.plot(ax,alpha=0.1)
                ax.set_title(f"Time-varying level set at t={t:.2f}")
        
        plt.show()

            


    
    def get_barrier_as_time_varying_polytopes(self):
        return self._time_varying_polytope_constraints

    def save_polytopes(self, filename):
        """
        Save a list of polytopes (H, b) with intervals to a file.
        
        Args:
            polytopes_list: List of tuples like [(H1, b1, (min1, max1)), (H2, b2, (min2, max2)), ...]
            filename: Output file path (e.g., 'polytopes.json')
        """
        
        for constraint in self._time_varying_polytope_constraints:
            constraint.to_file(filename)
 
            


class TimeVaryingConstraint:
    def __init__(self, start_time: float, end_time:float, H :np.ndarray, b:np.ndarray):
        
        
        self.start_time  :float       = start_time
        self.end_time    :float       = end_time
        self.H           : np.ndarray = H
        self.b           : np.ndarray = b

    def to_file(self, filename: str) -> None:
        """
        Save the time-varying constraint to a file.
        
        Args:
            filename: Output file path (e.g., 'constraint.json')
        """
        data = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'H': self.H.tolist(),
            'b': self.b.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def to_polytope(self) -> Polytope:
        """
        Convert the time-varying constraint to a Polytope object.
        
        Returns:
            Polytope object representing the time-varying constraint.
        """

        # add initial and final time constraints to the polytope
        row1 = np.hstack((np.zeros(self.H.shape[1]-1), np.array([1])))
        row2 = np.hstack((np.zeros(self.H.shape[1]-1), np.array([-1])))

        H_ext = np.vstack((self.H, row1, row2))
        b_ext = np.hstack((self.b, self.end_time, -self.start_time))

        # Plot the constraint as a polygon
        polytope = Polytope(H_ext, b_ext)


        return polytope
    
    def plot(self, ax: Optional[plt.Axes] = None) -> None:
        """
        Plot the time-varying constraint.
        
        Args:
            ax: Matplotlib Axes object to plot on. If None, create a new figure and axis.
        """
        if ax is None:
            fig, ax = plt.subplots()
        

        polytope = self.to_polytope()
        polytope.plot(ax)
        
        # Set title and labels
        ax.set_title(f'Time-Varying Constraint [{self.start_time}, {self.end_time}]')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
    
    

    
















