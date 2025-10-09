import numpy as np
import cvxpy as cp

from   typing     import Optional, Union
from   matplotlib import pyplot as plt
from   tqdm       import tqdm
import json
from scipy import sparse
from multiprocessing import Pool


from .logic         import (Formula, 
                           AndOperator,    
                           get_fomula_type_and_predicate_node, 
                           OrOperator,
                           UOp, FOp, GOp, 
                           Predicate, PredicateNode)


from .utils         import TimeInterval
from .linear_system import ContinuousLinearSystem,output_matrix, MultiAgentSystem

from ..polyhedron  import Polyhedron, BoxNd


class BarrierFunction :
    """

    This is an internal class that represents a general time-varying constraint of the form :math:`Dx + c+ \gamma(t) >=0`.
    
    
    We represent such constraint as a barrier function with multiple output as :math:`b(x,t) = Dx +c + \gamma(t)`, where 
    :math:`Dx + c >=0` represents a polytope. 

    The function gamma is a piexe wise linear function of the form :math:`\gamma(t) = e \cdot t + g`.
    """
    def __init__(self, polytope: Polyhedron, time_grid : list[float],flat_time : float) -> None:
        """
        Initialize the barrier function with a given polytope.

        :param polytope: Polytope object representing the constraint.
        :type polytope: Polytope
        """
        
        self.polytope  : Polyhedron  = polytope
        self.time_grid : list[float] = sorted(time_grid)
        
        self.time_deltas        : np.ndarray = np.array([time_grid[jj+1] - time_grid[jj] for jj in range(len(time_grid)-1)])
        self.num_time_intervals : int        = len(time_grid)-1
        self.flat_time          : float      = flat_time
        

        self.slopes_var  : cp.Variable = cp.Variable((polytope.num_hyperplanes,len(time_grid)-1), neg= True)# one slope for each interval
        self.gamma_0_var : cp.Variable = cp.Variable((polytope.num_hyperplanes), pos=True, name="gamma_0") # initial value of the gamma function
        self.r_var       : cp.Variable = cp.Variable( pos=True, name="robustness")  # robustness of the barrier function
        
        
        self.r_var.value = 100. # initial guess for the robustness

        self._D_high_order = None
        self._c_high_order = None
        
        self.flat_time : float = flat_time # time from which the slope must be zero
    
    
    @property
    def D(self)-> np.ndarray:
        """
        D matrix of the barrier function :math:`b(x,t) = Dx +c + \gamma(t)`
        """
        D = - self.polytope.A 
        return D
    @property
    def c(self) :
        """
        c vector of the barrier function :math:`b(x,t) = Dx +c + \gamma(t)`
        """
        return self.polytope.b 
    
    @property
    def D_high_order(self) -> np.ndarray:
        if self._D_high_order is None:
            return self.D
        else :
            return self._D_high_order
    @property
    def c_high_order(self)-> np.ndarray:
        if self._c_high_order is None:
            return self.c
        else :
            return self._c_high_order
        


    def e_vector(self, t:float) -> cp.Variable:
        """
        time derivative of the gamma function for a given section
        """

        if t < self.time_grid[0] or t > self.time_grid[-1]:
            raise ValueError("The time t is not in the time grid. Please check the time grid and the time t. The given time is " + str(t) + " and the time grid is " + str(self.time_grid))
        
        elif t == self.time_grid[-1] :
            e = self.slopes_var[:,-1]

        else : # if the time is in the range and it is not the last time in the grid then it will be contained in one of the intervals
            for jj in range(self.num_time_intervals):
                if t< self.time_grid[jj+1] and t >= self.time_grid[jj]:
                    e = self.slopes_var[:,jj]
                    break
        return e

    def g_vector(self,t:float) -> cp.Variable:
         # The piece wise affine model for the gamma function is given by
        # gamma_0 + sum_{j=0}^{n-1} slope_j * delta_t_j + e_n * (t - t_n).
        # where n represents the time interval within which the current time t is located.
        # This function return a vector of the form [delta_0, delta_1, ..., delta_n-1, -t_n,0,0,0 ...] that can be used 
        # to compute the value of gamma in matrix form as gamma_0 + slopes @ t_vector + e_n * t

        
        t_vector = np.zeros((self.num_time_intervals,))

        if t < self.time_grid[0] or t > self.time_grid[-1]:
            raise ValueError("The time t is not in the time grid. Please check the time grid and the time t. The given time is " + str(t) + " and the time grid is " + str(self.time_grid))
        
        elif t == self.time_grid[-1] :
            t_vector[:-1] = self.time_deltas[:-1]
            t_vector[-1]  = -self.time_grid[-2]
        
            
        else : # if the time is in the range and it is not the last time in the grid then it will be contained in one of the intervals
            for jj in range(self.num_time_intervals):
                if t< self.time_grid[jj+1] and t >= self.time_grid[jj] :
                    
                   
                    t_vector[:jj] = self.time_deltas[:jj]
                    t_vector[jj]  = -self.time_grid[jj]
                    break  
                
        g = self.gamma_0_var + self.slopes_var @ t_vector
        
        return g

    
    def gamma_at_time(self,t:float) -> cp.Variable:
        """
        Gamma function at a given time t.
        """


        return self.e_vector(t)*t + self.g_vector(t)
    
    def get_gamma_flat_constraint(self):
        
        constraints = []

        if self.flat_time == self.time_grid[-1]:
            # it is an eventually task and so there is not flat region of the gamma
            gamma_flat = self.gamma_at_time(self.flat_time) 
            constraints += [gamma_flat == -self.r_var] # eual to the robustness
        
        else:

            # Find the indices jj where time_grid[jj] >= flat_time
            flat_start_index = np.where(self.time_grid >= self.flat_time)[0][0]  # first index jj such that time_grid[jj] >= flat_time

            # Get the relevant time points and slopes
            flat_time_points = self.time_grid[flat_start_index:]
            flat_slopes      = self.slopes_var[:, flat_start_index:]  # shape: (dim, num_flat_intervals)
            

            # Apply the flat slope constraint (slopes must be 0)
            constraints += [flat_slopes == 0]
            constraints += [self.gamma_at_time(flat_time_points[0])== -self.r_var] # just set the firt flat time to -r and the other ones will be naturally equal to -r for the flat constraints
        
        return constraints
    

    def plot_gamma(self, ax,**kwargs) -> None:
        if self.gamma_0_var is None or self.slopes_var is None:
            print("Barrier yet not optimized. Please optimize the barrier first.")
        
        if ax is None :
            fig, ax = plt.subplots(figsize=(10, 4))

        gamma_line = []
        time_range = np.linspace(self.time_grid[0],self.time_grid[-1],100)
        for t in time_range:
            gamma_line += [self.gamma_at_time(t).value]
        
        gamma_line = np.array(gamma_line)
        label_plot = kwargs.pop("label",None)
        for jj in range(gamma_line.shape[1]) :
            if jj == 0:
                ax.plot(time_range, np.array(gamma_line[:,jj]), label = label_plot, **kwargs)
            else :
                ax.plot(time_range, np.array(gamma_line[:,jj]), **kwargs)

        ax.set_title("Gamma function")
        ax.set_xlabel("Time")
        ax.set_ylabel("Gamma")


    
    def set_high_order_constraints(self, D_high: np.ndarray, c_high: np.ndarray) -> None:
        """
        A barrier function is associated with an order depending on the dynamics it is applied to. Namely, a common constraints applied to defined the forward invariance of the \n
        barrier function is given as :math:`\dot{b}(x,t) + k\cdot b(x,t) >= 0` where k is a postive gain. For a linear system we have that 
        
        .. math::
            \dot{b}(x,t) = D  (Ax+Bu)  + \dot{\gamma(t)} + k(Dx + c + \gamma(t)).

        In case there one entry of the matrix :math:`D\cdot B` that is a vector of all zeros, then this direction is uncontrollable. For this reason we define high-order barrier functions \n
        by lifting the barrier function according to the following recursion 
        
        :math:`b^{p}(x,t) = D(A + kI)^{p} + \gamma(t) + k^{p}c

        where the recursion continues until math:`D(I + kc)^{p}B` has not direction of full zeros. If the system is controllable then this will happen after at most a number of iterations equal to the state space dimension.


        :param D_high: High-order D matrix.
        :type D_high: np.ndarray
        :param c_high: High-order c vector.
        :type c_high: np.ndarray
        """
        self._D_high_order = D_high
        self._c_high_order = c_high



class AlwaysTask:
    """
    This is an interval reprentation of a task. It is just a disctionary that contains a polytope (the predicate level set of the formula),
    The time interval of satisfaction (obtained by the task scheduler) and the type of the task.
    """
    def __init__(self, polytope: Polyhedron,time_interval:TimeInterval, secondary_time_interval : TimeInterval = None) -> None:
        self.polytope  : Polyhedron   = polytope
        self.alpha_var : cp.Variable =  cp.Variable(nonneg=True)  # stores the alpha value of the barrier.
        self.beta_var  : cp.Variable =  cp.Variable(nonneg=True)  # stores the beta value of the barrier.
        self.time_interval : TimeInterval = time_interval # time interval of the task
        self.secondary_time_interval : TimeInterval = secondary_time_interval # time interval of the task
    

class EventuallyTask:
    """
    This is an interval reprentation of a task. It is just a disctionary that contains a polytope (the predicate level set of the formula),
    The time interval of satisfaction (obtained by the task scheduler) and the type of the task.
    """
    def __init__(self, polytope: Polyhedron,time_interval:TimeInterval) -> None:
        self.polytope  : Polyhedron   = polytope
        self.beta_var  : cp.Variable  =  cp.Variable(nonneg=True)  # stores the alpha value of the barrier.
        self.time_interval : TimeInterval = time_interval # time interval of the task
        self.derived = False
    
    def flag_as_derived_from_always_eventually(self) :
        """
        Flag the task as derived from an always-eventually task.
        """
        self.derived = True

    


class TaskScheduler:
    def __init__(self,formula : Formula, systems : ContinuousLinearSystem) -> None:

        self.formula : Formula                = formula
        self.system  : ContinuousLinearSystem = systems

        self.tasks_list      : list[EventuallyTask|AlwaysTask] = [] # list of tasks
        self.varphi_tasks    : list[Formula]  = [] # list of varphi tasks (only always and eventually tasks here)

        self.already_optimized : bool = False
        self.time_constraints :list[cp.Constraint]   =  []

        self.get_varphi_tasks()
        self.detect_conflicting_conjunctions_and_give_good_time_guesses()
        self.make_time_schedule()

    def get_varphi_tasks(self) :
        """
        From this function, the following tasks are accomplished:
            1) The given formula is sectioned into subtasks varphi of type F,G,FG,GF.
            2) For each formula we create the time variables that are needed to satisfy the task.
        """

        is_a_conjunction = False

        # Check that the provided formula is within the fragment of allowable formulas.
        if isinstance(self.formula.root,OrOperator): #1) the root operator can be an or, but it not implemented for now.
            raise NotImplementedError("OrOperator is not implemented yet. Wait for it. it is coming soon.")
        elif isinstance(self.formula.root,AndOperator): # then it is a conjunction of single formulas
            is_a_conjunction = True
        else: #otherwise it is a single formula
            pass
        
        # subdivide in sumbformulas
        potential_varphi_tasks : list[Formula] = []
        if is_a_conjunction :
            for child_node in self.formula.root.children : # take all the children nodes and check that the remaining formulas are in the predicate
                varphi = Formula(root = child_node)
                potential_varphi_tasks += [varphi]
  
        else: 
            potential_varphi_tasks = [self.formula]

        for varphi in  potential_varphi_tasks : # take all the children nodes and check that the remaining formulas are in the predicate
            
            varphi_type, predicate_node = get_fomula_type_and_predicate_node(formula = varphi)

            root          : Union[FOp,UOp,GOp]   = varphi.root                  # Temporal operator of the fomula.
            time_interval : TimeInterval         = varphi.root.interval         # Time interval of the formula.
            
            polytope = Polyhedron(A = predicate_node.polytope.A, b = predicate_node.polytope.b) # bring polytope to suitable dimension
            task : AlwaysTask | EventuallyTask
            # Create barrier functions for each task
            
            if varphi_type == "G" :

                self.varphi_tasks += [varphi] # add the varphi task to the list of tasks
                task              = AlwaysTask(polytope,time_interval)
                self.tasks_list   += [task]

                # create time constraints
                self.time_constraints += [task.alpha_var == time_interval.a, task.beta_var == time_interval.b]
                
                # good initial guess 
                task.alpha_var.value = time_interval.a
                task.beta_var.value  = time_interval.b

            
            elif varphi_type == "F" :
                
                self.varphi_tasks += [varphi] # add the varphi task to the list of tasks
                task               = EventuallyTask(polytope, time_interval=time_interval)
                self.tasks_list   += [task]

                ## Give initial guess to the solver
                task.beta_var.value = time_interval.a

            
                # create time constraints
                self.time_constraints += [task.beta_var >= time_interval.a, 
                                          task.beta_var <= time_interval.b]
                

            elif varphi_type == "FG":

                G_operator          : GOp             = root.children[0]
                time_interval_prime : TimeInterval    = G_operator.interval # extract interval of the operator G
                
                self.varphi_tasks += [varphi] # add the varphi task to the list of tasks
                task               = AlwaysTask(polytope,
                                                time_interval= time_interval,
                                                secondary_time_interval= time_interval_prime)
                self.tasks_list   += [task]

                ## Give initial guess to the solver
                task.alpha_var.value = time_interval.get_sample() + time_interval_prime.a
                task.beta_var.value  = task.alpha_var.value + (time_interval_prime.b - time_interval_prime.a)

                # create time constraints
                self.time_constraints += [task.alpha_var >= time_interval.a +  time_interval_prime.a, 
                                          task.alpha_var <= time_interval.b +  time_interval_prime.a,
                                          task.beta_var  == task.alpha_var + (time_interval_prime.b - time_interval_prime.a)]
            


            elif varphi_type == "GF" : # decompose
                
                g_interval : TimeInterval = root.interval
                f_interval : TimeInterval = root.children[0].interval # the single children of the always operator is the eventually

                nf_min = np.ceil((g_interval.b - g_interval.a)/(f_interval.b - f_interval.a)) # minimum frequency of repetition
                m      = g_interval.a + f_interval.a

                interval       = g_interval.b - g_interval.a
                interval_prime = f_interval.b - f_interval.a
                delta_bar      = 1/nf_min *(interval/interval_prime)

                tau_prev = m

                for w in range(1,int(nf_min)+1):
                    
                    # keep order of defitiion as a_w_bar gets redefined
                    b_bar_w = m + w* 1        * interval_prime
                    a_bar_w = m + w* delta_bar* interval_prime
                    interval_w = TimeInterval(a = a_bar_w, b = b_bar_w)

                    self.varphi_tasks += [varphi] # add the varphi task to the list of tasks
                    task = EventuallyTask(polytope,time_interval=interval_w)
                    self.tasks_list   += [task]
                    task.flag_as_derived_from_always_eventually()

                    ## Give initial guess to the solver (this is just to speed up the solver)
                    task.beta_var.value =  m + w* delta_bar* interval_prime

                    # create time constraints
                    self.time_constraints += [task.beta_var >= tau_prev + delta_bar * interval_prime, 
                                              task.beta_var <= tau_prev  + 1         * interval_prime]
                    
                    tau_prev = task.beta_var # this is the link to the previous task.
                    
            else:
                raise ValueError("At least one of the formulas set in conjunction is not within the currently allowed grammar. Please verify the predicates in the formula.")


    def active_tasks_map(self,t: float) -> list[int]:
        """
        Returns the list of indices of active barriers at time t.
        """

        if self.already_optimized :
            active_tasks : list[int] = []
            for i, task in enumerate(self.tasks_list):
                if  t < task.beta_var.value:
                    active_tasks.append(i)
        else:
            raise ValueError("The tasks have not been optimized yet. Please optimize the tasks first.")

        
        return active_tasks
    


    def show_time_schedule(self) :
        
        tasks_schedules = []
        for i, task in enumerate(self.tasks_list):
             # add tasks to the list for plotting
            start_time = task.time_interval.a
            duration   =  task.time_interval.b - start_time
            varphi_type = "G" if isinstance(task, AlwaysTask) else "F"
            if isinstance(task, EventuallyTask) and task.derived:
                tasks_schedules.append({'start_time': start_time, 'duration': duration, 'type': varphi_type,"derived": True})
            else:
                tasks_schedules.append({'start_time': start_time, 'duration': duration, 'type': varphi_type,"derived": False})



        print("============================================================")
        print("Enumerating tasks")
        print("============================================================")
        
        for task in tasks_schedules:

            print("Found Tasks of type: ", task["type"])
            print("Start time: ", task["start_time"])
            print("Duration: ", task["duration"])
            if task["derived"]:
                print("(note*) Derived from an always-eventually task")
            print("---------------------------------------------")


        # plotting 
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot each task as a thin bar on its own y-row
        for i, schedule in enumerate(tasks_schedules):
            ax.broken_barh([(schedule["start_time"], schedule["duration"])], (i - 0.4, 0.8), facecolors='tab:blue')

        # Labeling
        ax.set_xlabel('Time')
        ax.set_ylabel("tasks")
        ax.set_yticks(range(len(tasks_schedules)))
        ax.set_yticklabels([rf'$\phi =  {schedule["type"]}  \mu$' for schedule  in tasks_schedules])
        ax.grid(True)


    

    
    def plot_active_set_map_cardinality(self) -> None:
        """
        Plot the result gamma functions for each task
        """

        if self.already_optimized:
            max_beta = 0.
            for task in self.tasks_list:
                max_beta = max(max_beta,task.beta_var.value)
            # plot the cardinality of the active set map
            fig, ax = plt.subplots(figsize=(10, 4))
            t = np.linspace(0, max_beta, int(max_beta*100))
            card = []
            for time in t:
                card += [len(self.active_tasks_map(time))]
            
            ax.plot(t, card, label='Cardinality of Active Set')
            ax.grid()
            ax.set_xlabel("Time")
            ax.set_ylabel("Number of active tasks")

            
            plt.tight_layout()
        else:
            pass

    

    
    def distance(self,p1:Polyhedron, p2: Polyhedron):
        """
        Use cvxpy to computer the distance between two polytopes inside two polytopes"
        """

        x = cp.Variable((self.system.state_dim))
        y = cp.Variable((self.system.state_dim))

        A1,b1 = p1.A, p1.b
        A2,b2 = p2.A, p2.b

        constraints = [A1@x <= b1, A2@y <= b2]
        cost        = cp.sum_squares(x-y)
        problem     = cp.Problem(cp.Minimize(cost), constraints)
        
        
        problem.solve("SCS", verbose=False)
        if problem.status != cp.OPTIMAL:
            raise Exception("Problem not solved correctly. Retuned problem status is " + str(problem.status))
        else:  
            return cost.value
    
    
    def detect_conflicting_conjunctions_and_give_good_time_guesses(self)-> None :
        """ This check can be expensive if you have many tasks. But it can be useful to avoid problems"""
        
        
        always_tasks : list[AlwaysTask] = [task for task in  self.tasks_list if isinstance(task,AlwaysTask) and task.secondary_time_interval is None]
        eventually_tasks : list[EventuallyTask] = [task for task in  self.tasks_list if isinstance(task,EventuallyTask)]

        
        # phase 1 (check always-always conflict) : two always tasks must have intersecting predicates
        for task_i in always_tasks :
            for task_j in always_tasks :
                if task_i != task_j:

                    time_interval_i = task_i.time_interval
                    time_interval_j = task_j.time_interval
                    pi = task_i.polytope
                    pj = task_j.polytope

                    if time_interval_i / time_interval_j is not None:
                        if self.distance(pi,pj) > 1E-4:
                            message = (f"Found conflicting conjunctions: A task of type G_[a,b]\mu is conflicting with a task of type G_[a',b']\mu" +
                                        "The interval of the tasks is intersectinig ")
                            raise Exception(message)
                        
                    
        # phase 2 (check always-eventually conflict) : if an eventually task is bounded inside an always task then the two must be intersecting     
        for task_e in eventually_tasks :
            for task_a in always_tasks :
                time_interval_e = task_e.time_interval
                time_interval_a = task_a.time_interval
                pe = task_e.polytope
                pa = task_a.polytope
                
                # If the eventually time interval is within the always time interval then you will have a conflict
                if time_interval_e in time_interval_a:
                    if self.distance(pe,pa) > 0.:
                        message = (f"Found conflicting conjunctions: A task of type F_[a,b]\mu is conflicting with a task of type G_[a',b']\mu" +
                                    "The eventually task a time interval contained within an always task but the predicates do not intersect. ")
                        raise Exception(message)
                    
                else : # try to iniitalize the eventually task better
                    if time_interval_e.a <= time_interval_a.a:
                        task_e.beta_var.value  = time_interval_e.a 
                    elif time_interval_e.b >= time_interval_a.b:
                        task_e.beta_var.value  = time_interval_a.b
                
    def make_time_schedule(self) :
        """
        From this function, the following tasks are accomplished:
            1) The given formula is sectioned sub tasks varphi.
            2) Each task var phi is converted into a barrier function.
            3) A time schedule for each task by assigning proper satisfaction and conclusion times to each task based on the temporal operators of the task.
        """
        
        # create optimization problem 

        cost = 0
        normalizer = 100.
        lift_up_factor = 1000.
        
        time_instants      = [ task.alpha_var.value for task in self.tasks_list if hasattr(task,"alpha_var")] + [task.beta_var.value for task in self.tasks_list]
        time_instants_vars = [ task.alpha_var for task in self.tasks_list if hasattr(task,"alpha_var")] + [task.beta_var for task in self.tasks_list]
        
        # Zip and sort based on time_instants
        sorted_pairs = sorted(zip(time_instants, time_instants_vars), key=lambda pair: pair[0])

        # Extract sorted time_instants_vars
        sorted_time_instants_vars = [var for _, var in sorted_pairs]
        
        for jj in range(len(sorted_time_instants_vars)-1):
            cost += lift_up_factor * cp.exp(-(sorted_time_instants_vars[jj+1] - sorted_time_instants_vars[jj])/normalizer)
                    

        problem = cp.Problem(cp.Minimize(cost),  self.time_constraints)
        problem.solve(warm_start=True, verbose=False, solver=cp.MOSEK)
        print("===========================================================")
        print("Times Schedule completed")
        print("===========================================================")
        print("Number of tasks created: ", len(self.tasks_list))
        print("Times optimization status: ", problem.status)
        
        print("Listing alpha and beta values per task :")
        for task in self.tasks_list:  
            if isinstance(task,AlwaysTask):
                print("Operator   : ", "Always")
                print("Time alpha : ", task.alpha_var.value)
                print("Time beta  : ", task.beta_var.value)
                print("-----------------------------------------------------")
            else:
                print("Operator   : ", "Eventually")
                print("Time tau   : ", task.beta_var.value)
                if task.derived:
                    print("(note*) Derived from an always-eventually task")
                print("-----------------------------------------------------")

        
        self.already_optimized = True


class BarriersOptimizer:
    
    def __init__(self, tasks_list          : list[AlwaysTask|EventuallyTask], 
                       workspace           : Polyhedron, 
                       system              : ContinuousLinearSystem,
                       input_bound         : Polyhedron ,
                       x_0                 : np.ndarray,
                       minimize_robustness : bool = True,
                       k_gain              : float = -1.,
                       solver              : str = "MOSEK",
                       relax_input_bounds  : bool = False) -> None:

        self.tasks_list        : list[AlwaysTask|EventuallyTask] = tasks_list
        self.workspace         : Polyhedron                      = workspace
        self.system            : ContinuousLinearSystem          = system
        self.input_bounds      : Polyhedron                      = input_bound
        self.x_0               : np.ndarray                      = x_0
        self.minimize_r        : bool                            = minimize_robustness
        self.given_k_gain      : float                           = k_gain
        self.robustness        : float                           = 0. # initial guess for the robustness
        self.solver            : str                             = solver
        self.relax_input_bounds: bool                            = relax_input_bounds

        self.barriers                          : list[BarrierFunction]       = []
        self.time_varying_polytope_constraints : list[TimeVaryingConstraint] = []

        list_of_switches = sorted(list({ float(task.alpha_var.value) for task in self.tasks_list if hasattr(task,"alpha_var")} | { float(task.beta_var.value) for task in self.tasks_list} | {0.} ))
        
        additional_switches = []
        # # add point in between every pair of points
        # for jj in range(len(list_of_switches)-1):
        #     if list_of_switches[jj] != list_of_switches[jj+1]:
        #         additional_switches += [list_of_switches[jj] + (list_of_switches[jj+1] - list_of_switches[jj])/2]

        self.time_grid = sorted(list_of_switches+ additional_switches )
        
        if workspace.is_open:
            raise ValueError("The workspace is an open Polyhedron. Please provide a closed polytope")
        
        self.create_barriers()
        self.optimize_barriers()
        

    def create_barriers(self) :
        """
        From this function, the following tasks are accomplished:
            1) The given formula is sectioned sub tasks varphi.
            2) Each task var phi is converted into a barrier function.
        """

   
        for task in self.tasks_list:
            time_grid_barrier = [time for time in self.time_grid if  time <= task.beta_var.value]
            if hasattr(task,"alpha_var"):
                barrier = BarrierFunction(polytope = task.polytope, time_grid = time_grid_barrier, flat_time = task.alpha_var.value)
            else:
                barrier = BarrierFunction(polytope = task.polytope, time_grid = time_grid_barrier, flat_time = task.beta_var.value)

            self.barriers.append(barrier)

    
    def _make_high_order_corrections(self, system : ContinuousLinearSystem, k_gain : cp.Parameter) -> int:

        """
        A barrier function is associated with an order depending on the dynamics it is applied to. Namely, a common constraints applied to defined the forward invariance of the \n
        barrier function is given as :math:`\dot{b}(x,t) + k\cdot b(x,t) >= 0` where k is a postive gain. For a linear system we have that 
        
        .. math::
            \dot{b}(x,t) = D  (Ax+Bu)  + \dot{\gamma(t)} + k(Dx + c + \gamma(t)).

        In case there one entry of the matrix :math:`D\cdot B` that is a vector of all zeros, then this direction is uncontrollable. For this reason we define high-order barrier functions \n
        by lifting the barrier function according to the following recursion 
        
        :math:`b^{p}(x,t) = D(A + kI)^{p} + \gamma(t) + k^{p}c

        where the recursion continues until math:`D(I + kc)^{p}B` has not direction of full zeros. If the system is controllable then this will happen after at most a number of iterations equal to the state space dimension.
        
        :param system: The system to be used for the optimization.
        :type system: ContinuousLinearSystem
        :param k_gain: The gain parameter for the optimization.
        :type k_gain: cp.Parameter
        :return: The order of the barrier function.
        :rtype: int
        :raises ValueError: If the system is not controllable.
        """

        if not system.is_controllable():
            raise ValueError("The provided system is not controllable. As of now, only controllable systems can be considered for barriers optimization.")
        

        A = system.A 
        B = system.B
        I = np.eye(system.state_dim)

        print("================================================")
        print("Correcting barriers for high order systems")
        print("================================================")
        for barrier in self.barriers :
            D            = barrier.D
            c            = barrier.c
            right_order  = False
            order        = 0

            while not right_order : #(for controllable systems this will stop after at most a number of iterations equal to the state space)
                db = D@B
                # if there is at least one row of all zeros then you must go on with the orders
                any_uncontrollable_direction = False
                for jj in range(db.shape[0]):
                    if np.all(db[jj,:] == 0):
                        any_uncontrollable_direction = True
        
                if not any_uncontrollable_direction :
                    right_order = True
                else :
                    order += 1
                    D = D@(A + I)
            

            if order == 0:
                print(f"Found barrier function of order: {order}")
                barrier.set_high_order_constraints(D,c)
            else:
                print(f"Found barrier function of order: {order}")
                D_high_order = barrier.D@cp.power(A + I*k_gain,order) 
                c_high_order = cp.power(k_gain,order) * c
                barrier.set_high_order_constraints(D_high_order, c_high_order)
            
        return order


    def active_barriers_map(self,t: float) -> list[int]:
        """
        Returns the list of active barriers at time t.
        """
        active_barriers = []
        for i, barrier in enumerate(self.barriers):
            beta = barrier.time_grid[-1]
            if  t < beta :
                active_barriers.append(i)
        return active_barriers
    

    def optimize_barriers(self) :
        
        
        if self.input_bounds.is_open:
            raise ValueError("The input bounds are an open Polyhedron. Please provide a closed polytope")
        
        if self.input_bounds.num_dimensions != self.system.input_dim:
            raise ValueError("The input bounds polytope must be in the same dimension as the system input. Given input bounds are in dimension " +
                             str(self.input_bounds.num_dimensions) + ", while the system input dimension is " + str(self.system.input_dim))

        if not isinstance(self.system, (ContinuousLinearSystem, MultiAgentSystem)):
            raise ValueError("The system must be a ContinuousLinearSystem or MultiAgentSystem.")

        x_0 = self.x_0.flatten()
        if len(x_0) != self.workspace.num_dimensions:
            raise ValueError("The initial state must be a vector of the same dimension as the workspace.")

        
        vertices        = self.workspace.vertices.T # matrix [dim x,num_vertices]
        V               = np.hstack((vertices , vertices)) # vertices matrix for each interval
        x_dim           = self.workspace.num_dimensions
        num_vertices    = vertices.shape[1]
        num_intervals = len(self.time_grid) - 1 # number of time intervals in the time grid

        # integrate the alpha and beta switching times into the time grid (so time steps between switching times will not be homogenous)
        k_gain     = cp.Parameter(pos=True) #! (when program fails try to increase control input and increase the k_gain. Carefully analyze the situation. Usually it should be at leat equal to 1.)
        order      = self._make_high_order_corrections(system = self.system, k_gain = k_gain)
        

        A             = self.system.A
        B             = self.system.B
        slack         = cp.Variable(nonneg = True)
        slack_penalty = 1E6
        
        UU = cp.Variable((self.system.input_dim, 2*num_vertices*num_intervals))  


        constraints  :list[cp.Constraint]  = []

        # Input constraints.
        constraints += [sparse.csc_matrix(self.input_bounds.A) @ UU <= self.input_bounds.b.reshape(-1,1)] # input constraints for all vertices at time s_j

        # dynamic constraints (Note : this is the most computationally intensive constraint)
        if not self.relax_input_bounds:
            print("Adding Dynamic Inoput Constraints")
            for jj in tqdm(range(num_intervals)): # for each interval
                
                # Get two consecutive time intervals in the sequence.
                s_j        = self.time_grid[jj]
                s_j_plus_1 = self.time_grid[jj+1]-0.0001 # it is like the value at the limit
                
                U_j            = UU[:,jj*2*num_vertices:(jj+1)*2*num_vertices]     
                dyn            = (A @ V + B @ U_j)  
                
                
                # Forward-invariance constraints. 

                ee = []
                DD = []
                cc = []
                gamma_gamma = []
                
                for active_task_index in self.active_barriers_map(s_j): # for each active task
                    
                    barrier       = self.barriers[active_task_index] 
                    
                    D_high_order  = barrier.D_high_order 
                    c_high_order  = barrier.c_high_order.reshape((-1,1))

                    gamma_s_j        = cp.hstack([barrier.gamma_at_time(s_j).reshape((-1,1))]*num_vertices) 
                    gamma_s_j_plus_1 = cp.hstack([barrier.gamma_at_time(s_j_plus_1).reshape((-1,1))]*num_vertices)   
                    gamma            = cp.hstack([gamma_s_j, gamma_s_j_plus_1]) 
                    
                    e_s_j            = cp.hstack([barrier.e_vector(s_j).reshape((-1,1))]*num_vertices) 
                    e_s_j_plus_1     = cp.hstack([barrier.e_vector(s_j_plus_1).reshape((-1,1))] * num_vertices) 
                    e                = cp.hstack([e_s_j, e_s_j_plus_1]) 
                    
                    ee.append(e)
                    DD.append(D_high_order)
                    cc.append(c_high_order)
                    gamma_gamma.append(gamma)

                ee    = cp.vstack(ee) # e vector for a given section
                DD    = cp.vstack(DD) # D matrix for a given section
                cc    = cp.vstack(cc) # c vector for a given section
                gamma_gamma = cp.vstack(gamma_gamma) # gamma function for a given section
                    
                constraints  += [DD @ dyn + ee + k_gain * (DD @ V + cc + gamma_gamma ) + slack >= 0]

        else:
            print("Skipping Dynamic Input Constraints")
                    

        # Add flatness constraint
        for barrier in self.barriers:
            constraints += barrier.get_gamma_flat_constraint()

        # Inclusion constraints
        betas        = list({ barrier.time_grid[-1] for barrier in self.barriers} | {0.}) # set inside the list removes duplicates if any.
        betas        = sorted(betas)
        
        epsilon    = 1E-3
        zeta_vars  = cp.Variable(( x_dim, len(betas)))
        
        # inclusion constraints for all vertices at time beta_0 (beta_0 is the first time of the barrier function)
        # constraints  += [self.workspace.A @   zeta_vars[:,1:] <= self.workspace.b.reshape(-1,1)] # inclusion constraints for all vertices at time beta_l (beta_l is the last time of the barrier function)
        
        for l in range(1,len(betas)):
            beta_l = betas[l]
            zeta_l = zeta_vars[:,l].reshape((-1,1)) # zeta at time beta_l

            gamma_at_beta_l = []
            D_at_beta_l     = []
            c_at_beta_l     = [] 

            for l_tilde in self.active_barriers_map(betas[l-1]) : # barriers active at lim t-> - beta_l is equal to the one active at time beta_{l-1}
                gamma_at_beta_l.append(self.barriers[l_tilde].gamma_at_time(beta_l).reshape((-1,1)) )# gamma function for a given section
                D_at_beta_l.append(self.barriers[l_tilde].D  )
                c_at_beta_l.append(self.barriers[l_tilde].c.reshape((-1,1)))# c vector

            gamma_at_beta_l = cp.vstack(gamma_at_beta_l)                   # gamma function for a given section
            D_at_beta_l     = cp.vstack(D_at_beta_l)    # D matrix
            c_at_beta_l     = cp.vstack(c_at_beta_l)                       # c vector   
            
            constraints += [D_at_beta_l @ zeta_l + c_at_beta_l +  gamma_at_beta_l >= epsilon ] # epsilon just to make sure the point is not at the boundary and it is strictly inside
        
        # set the zeta at beta=0 zero and conclude
        # initial state constraint
        zeta_0  = zeta_vars[:,0]

        D_0     = cp.vstack([barrier.D for barrier in self.barriers])
        c_0     = cp.hstack([barrier.c for barrier in self.barriers])
        gamma_0 = cp.hstack([barrier.gamma_0_var for barrier in self.barriers])

        constraints += [D_0 @ zeta_0 + c_0 + gamma_0 >= epsilon]

        
        # initial state constraint
        constraints += [zeta_0 == x_0]

        # create problem and solve it
        cost = 0.
        if self.minimize_r:
            all_r = cp.hstack([barrier.r_var for barrier in self.barriers])  # shape (d, K)
            cost += -cp.sum(all_r)
            
            slack_cost = slack_penalty * slack
            problem    = cp.Problem(cp.Minimize(cost+slack_cost), constraints)
        else:
            
            slack_cost = slack_penalty * slack
            problem    = cp.Problem(cp.Minimize(cost+slack_cost), constraints)


        if self.given_k_gain < 0.:
            good_k_found = False

            print("Selecting a good gain k ...")
            # when barriers have order highr than 1, the problem is no more dpp and thus it takes a lot of time to solve it.
            if order > 1:
                k_vals = np.arange(0.000001, 0.5, 0.03)
            else :
                k_vals = np.arange(0.001, 1, 0.003)
            
            best_k    = k_vals[0]
            best_slak = 1E10
            # Parallelize using multiprocessing
            with tqdm(total=len(k_vals)) as pbar:
                for k_val in k_vals:
                    pbar.set_description(f"k = {k_val:.3f}")
                    k_gain.value = k_val
                    
                    try :
                        problem.solve(warm_start=True, verbose=False, solver=self.solver)
                        pbar.update(1)
                    except Exception as e:
                        pbar.update(1)
                        continue
                    if problem.status == cp.OPTIMAL and slack.value < 1E-5:
                        best_k = k_val
                        good_k_found = True
            

                    elif problem.status == cp.OPTIMAL and slack.value > 1E-5 and not good_k_found:
                        if slack.value <= best_slak :
                            best_slak = slack.value
                            best_k    = k_val 
                    else :
                        continue
                

            if not good_k_found:
                print("No good k found. Please increase the range of k. Returing k with minimum violation")
                k_gain.value = best_k
                problem.solve(warm_start=True, verbose=False, solver=self.solver)
            else:
                k_gain.value = best_k
                problem.solve(warm_start=True, verbose=False,solver=self.solver )
        else:
            print("Given k_gain:",self.given_k_gain)
            k_gain.value = self.given_k_gain

            try :
                problem.solve(warm_start=True, verbose=True, ignore_dpp = True,solver=self.solver)
            except Exception as e :
                print(f"Error in solving the problem with given k_gain {self.given_k_gain}. The error is the following")
                raise e

        print("===========================================================")
        print("Barrier functions optimization result")
        print("===========================================================")
        print("Status                         : ", problem.status)
        print("Solver time                    : ", problem.solver_stats.solve_time)
        print("Number of variables            : ", sum(var.size for var in problem.variables()))
        print("Optimal Cost (expluding slack) :" , cost.value if hasattr(cost,"value") else cost)
        print("Maximum Slack violation        : ", slack.value)
        print('Robustness                     : ', min( np.min(barrier.r_var.value) for barrier in self.barriers))
        print('K gain                         : ', k_gain.value)
        print("-----------------------------------------------------------")
        print("Listing parameters per task")

        for task,barrier in zip(self.tasks_list,self.barriers) :
            print("Task Type       :", "Eventually" if isinstance(task,EventuallyTask) else "Always")
            print("Task Interval   :", task.time_interval)
            if isinstance(task,EventuallyTask):
                print("Task tau        :", task.beta_var.value)
            print("Barrier gamma_0 : ", barrier.gamma_0_var.value)
            print("Barrier r       : ", barrier.r_var.value)
            print("---------------------------------------------------")

        if problem.status != cp.OPTIMAL:
            print("Problem is not optimal. Terminate!")
            exit()
        
        # saving constraints as time_state constraints
        for barrier in self.barriers:
            D = barrier.D
            c = barrier.c

            for jj,time in enumerate(barrier.time_grid[:-1]) :
                
               
                e = barrier.e_vector(time).value
                g = barrier.g_vector(time).value

                start = barrier.time_grid[jj]
                end   = barrier.time_grid[jj+1]
                
                # convert from the form Dx + c >= 0 to Hx <= b

                H = - np.hstack((D,e[:,np.newaxis]))
                b =  c + g

                
                if np.abs((end-start))>= 1E-5: # avoid inserting the polytope if the interval of time is very small
                    self.time_varying_polytope_constraints += [TimeVaryingConstraint(start_time=start, end_time=end, H=H, b=b)] # the second part of the constraint it is just flat so we can remove it.
        self.robustness = min( np.min(barrier.r_var.value) for barrier in self.barriers) # get the minimum robustness of the barriers
        return problem.solver_stats
    
    def get_robustness(self) -> float:
        """
        Returns the minimum robustness of the barriers.
        """
        return self.robustness


    def plot_gammas(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Number of colors you want
        num_colors = len(self.barriers)
        # Get the Viridis colormap
        cmap = plt.get_cmap("tab10")
        # Pick random values between 0 and 1
        colors = [cmap(val) for val in np.linspace(0, 1, num_colors)]


        for jj,barrier in enumerate(self.barriers):
            barrier.plot_gamma(ax,c=colors[jj], label = fr"$\gamma_{jj}(t)$")
            
        
        ax.axhline(0, color='black', linestyle='--', label = "Zero level")
        ax.set_title(r"$\gamma(t)$")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"$\gamma(t)$")
        ax.grid()
        ax.legend()


    def show_time_varying_level_set(self, ax = None,t_start :float = 0., t_end :float = 1., n_points :int = 10) :
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            self.workspace.plot(ax, alpha=0.01) # plot transparent workspace to set the right figure size
        
        # for now only 2d workspaces can be plotted. Extensions will come soon.
        if self.workspace.num_dimensions != 2:
            raise ValueError("The workspace must be in 2D. The current implementation only supports 2D plotting.")
        
        color = np.random.rand(3,)
        for t in np.linspace(t_start, t_end, n_points):
            
            polytopes = []
            for constrain in self.time_varying_polytope_constraints:
                if t <= constrain.end_time and t >= constrain.start_time:
                    H = constrain.H[:,:-1]
                    b = constrain.b
                    e = constrain.H[:,-1]

                    polytope = Polyhedron(H, b - e*t)
                    polytopes.append(polytope)

            # create intersections
            if len(polytopes) > 0:
                intersection : Polyhedron = polytopes[0]
                for i in range(1, len(polytopes)):
                    intersection = intersection.intersect(polytopes[i])
                
                intersection.plot(ax,alpha=0.1, color = color)

            
    def get_barrier_as_time_varying_polyhedrons(self):
        return self.time_varying_polytope_constraints

    def save_polytopes(self, filename :str):
        """
        Save a list of polytopes (H, b) with intervals to a file.
        
        Args:
            polytopes_list: List of tuples like [(H1, b1, (min1, max1)), (H2, b2, (min2, max2)), ...]
            filename: Output file path (e.g., 'polytopes.json')
        """
        data = []
        for constraint in self.time_varying_polytope_constraints:
            data.append(constraint.as_dict())
        
        with open(filename, 'w') as f:
            json.dump(data, f)




def compute_polyhedral_constraints( formula      : Formula, 
                                    workspace    : Polyhedron, 
                                    system       : ContinuousLinearSystem|MultiAgentSystem, 
                                    input_bounds : Polyhedron, 
                                    x_0          : np.ndarray,
                                    plot_results : bool = False,
                                    k_gain       : float = -1.,
                                    solver       : str = "CLARABEL",
                                    relax_input_bounds : bool = False) -> tuple[list["TimeVaryingConstraint"],float]:


    """
    :param formula: The formula to be optimized.
    :type formula: Formula
    :param workspace: The workspace polytope.
    :type workspace: Polyhedron
    :param system: The system to be used for the optimization.
    :type system: ContinuousLinearSystem
    :param input_bounds: The input bounds polytope.
    :type input_bounds: Polyhedron
    :param x_0: The initial state of the system.
    :type x_0: np.ndarray
    
    """

    
    scheduler = TaskScheduler(formula, system) # create task optimizer


    barrier_optimizer = BarriersOptimizer(tasks_list   = scheduler.tasks_list, 
                                          workspace    = workspace, 
                                          system       = system,
                                          input_bound  = input_bounds,
                                          x_0          = x_0,
                                          k_gain       = k_gain,
                                          solver       = solver,
                                          relax_input_bounds=relax_input_bounds) # create barrier optimizer
    
    if plot_results :
        scheduler.show_time_schedule()
        scheduler.plot_active_set_map_cardinality()
        barrier_optimizer.plot_gammas()
    
    time_varying_constraints : list["TimeVaryingConstraint"] = barrier_optimizer.get_barrier_as_time_varying_polyhedrons()
    robustness :float = barrier_optimizer.get_robustness()
    
    return time_varying_constraints, robustness
            

class TimeVaryingConstraint:
    """
    Representation of time-varying constraints in the form of H [x,t]  <= b.
    The matrix H has dimension (p x dim+1) where p is the number of constraints
    and dim is the state dimension. For a single integrator in 2D the state dimension
    is 2 and thus the matrix H has dimension (p x 3) where the +1 is due to the 
    presence of the time dimension. For a double integrator in 3D the state dimension
    is 3 and thus the matrix H has dimension (p x 4). 
    """

    def __init__(self, start_time: float, end_time:float, H :np.ndarray, b:np.ndarray):
        
        
        self.start_time  :float       = start_time
        self.end_time    :float       = end_time
        self.H           : np.ndarray = H
        self.b           : np.ndarray = b
        self.system_dim  : int        = H.shape[1] - 1  # Last column is time, so subtract 1 for state dimensions
        self.p           : int        = H.shape[0]  # Number of constraints

    def as_dict(self) -> dict:
        """
        Convert the time-varying constraint to a dictionary.
        
        Returns:
            Dictionary representation of the time-varying constraint.
        """
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'H': self.H.tolist(),
            'b': self.b.tolist()
        }

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
    
    def to_polytope(self) -> Polyhedron:
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
        polytope = Polyhedron(H_ext, b_ext)


        return polytope
    
    def plot3d(self, ax: Optional[plt.Axes] = None,**kwords) -> None:
        """
        Plot the time-varying constraint.
        
        Args:
            ax: Matplotlib Axes object to plot on. If None, create a new figure and axis.
        """
        

        polytope = self.to_polytope()
        num_dims = polytope.num_dimensions

        if num_dims == 3: # 2d polytope + time
        
            if ax is None:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

            polytope.plot(ax, **kwords)
            
            # Set title and labels
            ax.set_title(f'Time-Varying Constraint [{self.start_time}, {self.end_time}]')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        elif num_dims >= 3: # 3d polytope + time
            if ax is None:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

            for t in np.linspace(self.start_time, self.end_time, 10):
                e = self.H[:,-1]
                b = self.b
                H = self.H[:,:3] # plot first three dimensions
                b_t = b - e*t
                polytope = Polyhedron(H, b_t)
                # Plot the constraint as a polygon
                polytope.plot(ax,**kwords)

            # Set title and labels
            ax.set_title(f'Time-Varying Constraint [{self.start_time}, {self.end_time}]')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            raise ValueError(f"Time-varying constraints can only be plotted in 3D or 4D. Given dimension: {num_dims}")


    def plot2d(self, ax: Optional[plt.Axes] = None) -> None:
        """
        Plot the time-varying constraint.
        
        Args:
            ax: Matplotlib Axes object to plot on. If None, create a new figure and axis.
        """
        

        polytope = self.to_polytope()
        num_dims = polytope.num_dimensions

        if num_dims != 3:
            raise ValueError("Time-varying constraints can only be plotted in 2D.")
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(10, 4))
        
        for t in np.linspace(self.start_time, self.end_time, 10):
            e = self.H[:,-1]
            b = self.b
            H = self.H[:,:-1]
            b_t = b - e*t
            polytope = Polyhedron(H, b_t)
            # Plot the constraint as a polygon
            polytope.plot(ax)

        

        # Set title and labels
        ax.set_title(f'Time-Varying Constraint [{self.start_time}, {self.end_time}]')
        ax.set_xlabel('x')
        ax.set_ylabel('y')