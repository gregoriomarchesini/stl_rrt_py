import numpy as np
import casadi as ca

from   typing     import Optional, Union
from   matplotlib import pyplot as plt
from   tqdm       import tqdm
from   time       import perf_counter
import json

from multiprocessing import Pool


from .logic         import (Formula, 
                           AndOperator,    
                           get_fomula_type_and_predicate_node, 
                           OrOperator,
                           UOp, FOp, GOp,)


from .utils         import TimeInterval
from .linear_system import ContinuousLinearSystem

from ..polyhedron  import Polyhedron


class BarrierFunction :
    """
    This is an internal class that represents a general time-varying constraint of the form :math:`Dx + c+ \gamma(t) >=0`.
    
    
    We represent such constraint as a barrier function with multiple output as :math:`b(x,t) = Dx +c + \gamma(t)`, where 
    :math:`Dx + c >=0` represents a polyhedron. 

    The function gamma is a piece-wise linear function of the form :math:`\gamma(t) = e \cdot t + g`, where :math:`e` 
    is a vector of slopes and :math:`g` is a constant offset.
    """
    def __init__(self, polyhedron: Polyhedron, time_grid : list[float], flat_time : float, opti: ca.Opti) -> None:
        """
        Initialize the BarrierFunction with a polyhedron (representing the level set of a predicate for a given STL task),
        a time grid (a list of time instants where the barrier function will decay in time), 
        and a flat time (the time after which the slope of the gamma function must be zero). The opti class will be needed for 
        optimizing the parameters of the barrier function.

        :param polyhedron: The polyhedron representing the level set of a predicate for a given STL task.
        :type polyhedron: Polyhedron
        :param time_grid: A list of time instants where the barrier function will decay in time.
        :type time_grid: list[float]
        :param flat_time: The time after which the slope of the gamma function must be zero.
        :type flat_time: float
        :param opti: The casadi Opti class used for optimization.
        :type opti: ca.Opti
        
        """
        
        self.polyhedron  : Polyhedron  = polyhedron         
        self.time_grid   : list[float] = sorted(time_grid)  # time grid over which the gamma funciton of the barrier will decrease piece-wise linearly. Over each interval gamma(t) will decay linearly
        self.opti        : ca.Opti     = opti               # the optimization problem
        
        self.time_step          : list[float] = [time_grid[jj+1] - time_grid[jj] for jj in range(len(time_grid)-1)]
        self.num_time_intervals : int         = len(time_grid)-1
        self.flat_time          : float       = flat_time # time after which the slope is constrained to be zero
        

        self.slopes_var  : list[ca.MX] = [self.opti.variable(polyhedron.num_hyperplanes) for jj in range(len(time_grid)-1) ]# one slope for each interva
        self.gamma_0_var : ca.MX = self.opti.variable(polyhedron.num_hyperplanes) # initial value of the gamma function
        self.r_var       : ca.MX = self.opti.variable()  # robustness of the barrier function
        
        self.opti.set_initial(self.r_var,100) # initial guess for the robustness

        self._D_high_order = None
        self._c_high_order = None
        
        self.flat_time : float = flat_time # time from which the slope must be zero
    
    
    @property
    def D(self)-> np.ndarray:
        """
        D matrix of the barrier function :math:`b(x,t) = Dx +c + \gamma(t)`
        """
        D = - self.polyhedron.A 
        return D
    
    @property
    def c(self) :
        """
        c vector of the barrier function :math:`b(x,t) = Dx +c + \gamma(t)`
        """
        return self.polyhedron.b 
    
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
        


    def e_vector(self, t:float) -> ca.MX:
        """
        time derivative of the gamma function for a given section
        """

        if t < self.time_grid[0] or t > self.time_grid[-1]:
            raise ValueError("The time t is not in the time grid. Please check the time grid and the time t. The given time is " + str(t) + " and the time grid is " + str(self.time_grid))
        
        elif t == self.time_grid[-1] :
            e = self.slopes_var[-1]

    
        else : # if the time is in the range and it is not the last time in the grid then it will be contained in one of the intervals
            for jj in range(self.num_time_intervals):
                if t< self.time_grid[jj+1] and t >= self.time_grid[jj] :
                    e = self.slopes_var[jj]
                    break
        return e

    def g_vector(self,t:float) -> ca.MX:
        """
        Constant offset of the gamma function for a given section. 
        """

        if t < self.time_grid[0] or t > self.time_grid[-1]:
            raise ValueError("The time t is not in the time grid. Please check the time grid and the time t. The given time is " + str(t) + " and the time grid is " + str(self.time_grid))

        elif t == self.time_grid[-1]:
            
            sum_gammas               = 0.
            for k in range(self.num_time_intervals-1) :
                sum_gammas += self.time_step[k] * self.slopes_var[k]
            
            
            
            one_before_the_last_time = self.time_grid[-2]
            last_slope               = self.slopes_var[-1]
            g                        = self.gamma_0_var + sum_gammas - last_slope*one_before_the_last_time
        
        else: # if the time is in the range and it is not the last time in the grid then it will be contained in one of the intervals

            for jj in range(self.num_time_intervals):
                if t< self.time_grid[jj+1] and t >= self.time_grid[jj] :
                    
                    if jj > 0:
                        sum_gammas = 0.
                        for k in range(jj):
                            sum_gammas += self.time_step[k] * self.slopes_var[k]

                    else :
                        sum_gammas= 0.
                
                    g = self.gamma_0_var + sum_gammas  - self.slopes_var[jj]*self.time_grid[jj] 
                    break
        
        return g
    
    def gamma_at_time(self,t:float) -> ca.MX:
        """
        Gamma function at a given time t.
        """
        
        return self.e_vector(t)*t + self.g_vector(t)
    
    def get_constraints(self) -> list[ca.MX]:
        
        constraints = []
        
        ##############################################################
        # Flatness constraint of the gamma function
        # The gamma function must be flat after the flat_time.
        ##############################################################
        if self.flat_time == self.time_grid[-1]:
            # it is an eventually task and so there is not flat region of the gamma
            gamma_flat = self.gamma_at_time(self.flat_time) 
            constraints += [gamma_flat == -self.r_var] # eual to the robustness
        
        else:
            
            
            for jj in range(self.num_time_intervals):
                if self.time_grid[jj]>= self.flat_time : # the interval is contained between flat_time and then end of the grid
                    slope = self.slopes_var[jj]
                    gamma_flat = self.gamma_at_time(self.time_grid[jj])

                    constraints += [gamma_flat == -self.r_var] # eual to the robustness
                    constraints += [slope == 0.] # flat output

        ##############################################################
        # Constraints on the gamma function
        # The gamma function must be non-negative and decreasing.
        ##############################################################

        constraints += [self.r_var > 0.] # robustness must be positive
        constraints += [self.slopes_var[k] <= 0. for k in range(len(self.slopes_var))] # slopes must be non-positive
        constraints += [self.gamma_0_var > 0.] # initial value of the gamma function must be non-negative

        return constraints
    
    

    def plot_gamma(self, ax,**kwargs) -> None:
        if self.gamma_0_var is None or self.slopes_var is None:
            print("Barrier yet not optimized. Please optimize the barrier first.")
        
        if ax is None :
            fig, ax = plt.subplots(figsize=(10, 4))

        gamma_line = []
        time_range = np.linspace(self.time_grid[0],self.time_grid[-1],100)
        for t in time_range:
            gamma_line += [self.opti.value(self.gamma_at_time(t))]
        
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
    Internal class to represent an always task in the STL formula for optimization purposes.
    
    """
    def __init__(self, opti          : ca.Opti,
                       polyhedron    : Polyhedron,
                       time_interval : TimeInterval, 
                       secondary_time_interval : TimeInterval = None) -> None:
        
        
        
        self.opti                    : ca.Opti      = opti
        self.polyhedron              : Polyhedron   = polyhedron
        self.alpha_var               : ca.MX        = self.opti.variable()    # stores the alpha value of the barrier.
        self.beta_var                : ca.MX        = self.opti.variable()    # stores the beta value of the barrier.
        self.time_interval           : TimeInterval = time_interval           # time interval of the task
        self.secondary_time_interval : TimeInterval = secondary_time_interval # time interval of the task
    
    @property
    def alpha_var_value(self) -> float:
        """
        Returns the value of the alpha variable.
        """
        return float(self.opti.value(self.alpha_var))
    
    @property
    def beta_var_value(self) -> float:
        """
        Returns the value of the beta variable.
        """
        return float(self.opti.value(self.beta_var))


class EventuallyTask:
    """
    Internal class to represent an eventually task in the STL formula for optimization purposes.
    """
    def __init__(self, opti          : ca.Opti,
                       polyhedron    : Polyhedron,
                       time_interval : TimeInterval) -> None:
        
        
        
        self.opti          : ca.Opti      = opti
        self.polyhedron    : Polyhedron   = polyhedron              
        self.beta_var      : ca.MX        = self.opti.variable()       # stores the beta value of the barrier.
        self.time_interval : TimeInterval = time_interval              # time interval of the task
        self.derived       : bool         = False                      # mark if derived from an always eventually formula

    @property
    def beta_var_value(self) -> float:
        """
        Returns the value of the beta variable.
        """
        return float(self.opti.value(self.beta_var))
    
    def flag_as_derived_from_always_eventually(self) :
        """
        Flag the task as derived from an always-eventually task.
        """
        self.derived = True

    
class TaskScheduler: 
    """
    This is an internal class applied to separate a formula into subtasks and to create a time schedule for each task. From the given time shedule
    a set of time-varying barrier functions will be created in order to satisfy the task.
    """
    
    def __init__(self, formula : Formula, systems : ContinuousLinearSystem) -> None:
        """

        :param formula: The STL formula to be optimized.
        :type formula: Formula
        :param systems: The continuous linear system to be used for the optimization.
        :type systems: ContinuousLinearSystem
        """
        
        self.opti    : ca.Opti                = ca.Opti() # optimization problem
        self.formula : Formula                = formula
        self.system  : ContinuousLinearSystem = systems

        self.tasks_list      : list[EventuallyTask|AlwaysTask] = [] # list of internally represented tasks
        self.varphi_tasks    : list[Formula]                   = [] # subformulas always/eventually extracted from the global formula

        self.already_optimized : bool          = False
        self.time_constraints  : list[ca.MX]   = []    # constraints on the time variables of the tasks

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
                
                varphi                 = Formula(root = child_node)
                potential_varphi_tasks += [varphi]
  
        else: 
            potential_varphi_tasks = [self.formula]

        for varphi in  potential_varphi_tasks : # take all the children nodes and check that the remaining formulas are in the predicate
            
            varphi_type, predicate_node = get_fomula_type_and_predicate_node(formula = varphi)

            root          : Union[FOp,UOp,GOp]   = varphi.root              # Temporal operator of the fomula.
            time_interval : TimeInterval         = varphi.root.interval     # Time interval of the formula.
            dims          : list[int]            = predicate_node.dims      # Dimensions over which the formula is applied.
            
            # create output matrix since the polyhedron is defined in the output space.
            # ACx <= b.
            try : 
                C   = self.system.output_matrix_from_dimension(dims) 
            except Exception as e:
                print(f"Error in output matrix creation. The error stems from the fact that the required dimension of one or more predicates in the formulas" +
                    f"is out of range for the system e.g. a predicate is enforcing a specification of the state with index 3 but the state dimension is only 2. " +
                    f"The raised exception is the following")
                raise e
            
            polyhedron = Polyhedron(A = predicate_node.polyhedron.A@C, b = predicate_node.polyhedron.b) # bring polyhedron to suitable dimension
            task : AlwaysTask | EventuallyTask
            # Create barrier functions for each task
            
            if varphi_type == "G" :

                self.varphi_tasks += [varphi] # add the varphi task to the list of tasks
                task              = AlwaysTask(opti          = self.opti , 
                                               polyhedron    = polyhedron ,
                                               time_interval = time_interval)
                self.tasks_list   += [task]

                # create time constraints
                self.time_constraints += [task.alpha_var == time_interval.a, task.beta_var == time_interval.b]
                self.time_constraints += [task.alpha_var >= 0., task.beta_var >=0.]                            # time instants sgould be positive
                
                self.opti.set_initial(task.alpha_var, time_interval.a) # give initial guess to the solver
                self.opti.set_initial(task.beta_var, time_interval.b)  # give initial guess to the solver

            elif varphi_type == "F" :
                
                self.varphi_tasks += [varphi] # add the varphi task to the list of tasks
                task               = EventuallyTask(opti          = self.opti,
                                                    polyhedron    = polyhedron, 
                                                    time_interval = time_interval)
                self.tasks_list   += [task]
            
                ## create time constraints
                self.time_constraints += [task.beta_var >= time_interval.a, 
                                          task.beta_var <= time_interval.b]
                self.time_constraints += [task.beta_var >= 0.] # time instant should be positive
                
                ## Give initial guess to the solver
                self.opti.set_initial(task.beta_var, time_interval.a) # give initial guess to the solver

            elif varphi_type == "FG":

                G_operator          : GOp             = root.children[0]
                time_interval_prime : TimeInterval    = G_operator.interval # extract interval of the operator G
                
                self.varphi_tasks += [varphi] # add the varphi task to the list of tasks
                task               = AlwaysTask(opti                    = self.opti,
                                                polyhedron              =  polyhedron,
                                                time_interval           = time_interval,
                                                secondary_time_interval = time_interval_prime)
                self.tasks_list   += [task]

                ## Give initial guess to the solver
                alpha_guess = time_interval.get_sample() + time_interval_prime.a
                self.opti.set_initial(task.alpha_var, alpha_guess ) # give initial guess to the solver
                self.opti.set_initial(task.beta_var, alpha_guess  + (time_interval_prime.b - time_interval_prime.a))  # give initial guess to the solver

                # create time constraints
                self.time_constraints += [task.alpha_var >= time_interval.a +  time_interval_prime.a, 
                                          task.alpha_var <= time_interval.b +  time_interval_prime.a,
                                          task.beta_var  == task.alpha_var + (time_interval_prime.b - time_interval_prime.a)]
                self.time_constraints += [task.alpha_var >= 0., task.beta_var >= 0.] # time instants should be positive
            


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
                    task               = EventuallyTask(opti          = self.opti,
                                                        polyhedron    = polyhedron,
                                                        time_interval = interval_w)
                    
                    self.tasks_list   += [task]
                    task.flag_as_derived_from_always_eventually()

                    ## Give initial guess to the solver (this is just to speed up the solver)
                    self.opti.set_initial(task.beta_var,m + w* delta_bar* interval_prime)

                    # create time constraints
                    self.time_constraints += [task.beta_var >= tau_prev + delta_bar * interval_prime, 
                                              task.beta_var <= tau_prev  + 1        * interval_prime]
                    self.time_constraints += [task.beta_var >= 0.] # time instant should be positive
                    
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
                if  t < self.opti.value(task.beta_var):
                    active_tasks.append(i)
        else:
            raise ValueError("The tasks have not been optimized yet. Please optimize the tasks first.")

        
        return active_tasks
    


    def show_time_schedule(self) :
        
        print("============================================================")
        print("Enumerating tasks")
        print("============================================================")
        # plotting 
        fig, ax                = plt.subplots(figsize=(10, 4))
        task_types : list[str] = []
        for i, task in enumerate(self.tasks_list):
             # add tasks to the list for plotting
            start_time  = task.time_interval.a
            duration    =  task.time_interval.b - start_time
            varphi_type = "G" if isinstance(task, AlwaysTask) else "F"
            task_types += [varphi_type]
            
            print("Found Tasks of type: ", varphi_type)
            print("Start time:          ", start_time)
            print("Duration:            ", duration)
            if varphi_type == "F" :
                if task.derived:
                    print("(note*) Derived from an always-eventually task")
            print("---------------------------------------------")

            ax.broken_barh([(start_time, duration)], (i - 0.4, 0.8), facecolors='tab:blue')

        # Labeling
        ax.set_xlabel('Time')
        ax.set_ylabel("tasks")
        ax.set_yticks(range(len(self.tasks_list)))
        ax.set_yticklabels([rf'\phi =  {task_type}  \mu' for task_type in task_types])
        ax.grid(True)


    def plot_active_set_map_cardinality(self) -> None:
        """
        Plot the result gamma functions for each task
        """

        if self.already_optimized:
            max_beta = 0.
            for task in self.tasks_list:
                max_beta = max(max_beta,self.opti.value(task.beta_var))
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
        Use cvxpy to computer the distance between two polyhedrons inside two polyhedrons"
        """
        
        opti = ca.Opti("conic")
        x    = self.opti.variable(self.system.size_state)
        y    = self.opti.variable(self.system.size_state)

        A1,b1 = p1.A, p1.b
        A2,b2 = p2.A, p2.b

        constraints = [ca.mtimes(A1,x) <= b1, ca.mtimes(A2,y) <= b2]
        cost        = ca.sumsqr(x-y)

        opti.subject_to(*constraints)
        
        opti.minimize(cost)
        opti.solver("osqp")
        opti.solve()
        
        dist = opti.value(cost)
        #todo: add exception for failure
        return dist
    
    
    def detect_conflicting_conjunctions_and_give_good_time_guesses(self)-> None :
        """ 
        Checks if the given tasks are given as conflicting conjuncitons. Note that only a simple subset of the possible conflicting conjunctions is 
        detected.
        """
        
        
        always_tasks     : list[AlwaysTask]     = [task for task in  self.tasks_list if isinstance(task,AlwaysTask) and task.secondary_time_interval is None]
        eventually_tasks : list[EventuallyTask] = [task for task in  self.tasks_list if isinstance(task,EventuallyTask)]

        
        # phase 1 (check always-always conflict) : two always tasks that have interesting time intervals must have intersecting predicates
        for task_i in always_tasks :
            for task_j in always_tasks :
                if task_i != task_j:

                    time_interval_i = task_i.time_interval
                    time_interval_j = task_j.time_interval
                    pi              = task_i.polyhedron
                    pj              = task_j.polyhedron

                    if time_interval_i / time_interval_j is not None: # if the intersection is not empty
                        if self.distance(pi,pj) > 0.:
                            message = (f"Found conflicting conjunctions: A task of type G_[a,b]\mu is conflicting with a task of type G_[a',b']\mu" +
                                        "The interval of the tasks is intersectinig ")
                            raise Exception(message)
                        
                    
        # phase 2 (check always-eventually conflict) : if an eventually task is bounded inside an always task then the two must be intersecting     
        for task_e in eventually_tasks :
            for task_a in always_tasks :
                time_interval_e = task_e.time_interval
                time_interval_a = task_a.time_interval
                pe              = task_e.polyhedron
                pa              = task_a.polyhedron
                
                # If the eventually time interval is within the always time interval then you will have a conflict
                if time_interval_e in time_interval_a:
                    if self.distance(pe,pa) > 0.:
                        message = (f"Found conflicting conjunctions: A task of type F_[a,b]\mu is conflicting with a task of type G_[a',b']\mu" +
                                    "The eventually task a time interval contained within an always task but the predicates do not intersect. ")
                        raise Exception(message)
                    
                else : # try to iniitalize the eventually task better
                    if time_interval_e.a <= time_interval_a.a:
                        self.opti.set_initial(task_e.beta_var, time_interval_e.a) # give initial guess to the solver
                    
                    elif time_interval_e.b >= time_interval_a.b:
                        self.opti.set_initial(task_e.beta_var,time_interval_a.b)
                
    def make_time_schedule(self) :
        """
        From this function, the following tasks are accomplished:
            1) The given formula is sectioned sub tasks varphi.
            2) Each task var phi is converted into a barrier function.
            3) A time schedule for each task by assigning proper satisfaction and conclusion times to each task based on the temporal operators of the task.
        """
        
        # create optimization problem 

        cost           = 0
        normalizer     = 100.
        lift_up_factor = 1000.
        
        time_instants      = ([ self.opti.value(task.alpha_var,self.opti.initial()) for task in self.tasks_list if hasattr(task,"alpha_var")] + 
                              [ self.opti.value(task.beta_var,self.opti.initial()) for task in self.tasks_list] )
        
        time_instants_vars = [ task.alpha_var for task in self.tasks_list if hasattr(task,"alpha_var")] + [task.beta_var for task in self.tasks_list]
        
        # Zip and sort based on time_instants
        sorted_pairs = sorted(zip(time_instants, time_instants_vars), key=lambda pair: pair[0])

        print("sorted time instants: ", [pair[0] for pair in sorted_pairs])

        # Extract sorted time_instants_vars
        sorted_time_instants_vars = [var for _, var in sorted_pairs]
        
        for jj in range(len(sorted_time_instants_vars)-1):
            cost += lift_up_factor * ca.exp(-(sorted_time_instants_vars[jj+1] - sorted_time_instants_vars[jj])/normalizer)
                    
        
        self.opti.subject_to(self.time_constraints) # add time constraints to the optimization problem
        self.opti.minimize(cost) # minimize the cost function
        self.opti.solver("ipopt")  # Set the solver for the optimization problem
        self.opti.solve()

        print("===========================================================")
        print("Times Schedule completed")
        print("===========================================================")
        print("Number of tasks created   : ", len(self.tasks_list))
        print("Times optimization status : ", self.opti.stats()["return_status"])
        
        print("Listing alpha and beta values per task :")
        print("-----------------------------------------------------")
        for task in self.tasks_list:  
            if isinstance(task,AlwaysTask):
                print("Operator   : ", "Always")
                print("Time alpha : ", self.opti.value(task.alpha_var))
                print("Time beta  : ", self.opti.value(task.beta_var))
                print("-----------------------------------------------------")
            else:
                print("Operator   : ", "Eventually")
                print("Time tau   : ", self.opti.value(task.beta_var))
                if task.derived:
                    print("(note*) Derived from an always-eventually task")
                print("-----------------------------------------------------")

        
        self.already_optimized = True


class BarriersOptimizer:
    """
    This class is used to optimize the parameters of the barrier functions for a given set of tasks.
    It creates a set of time-varying barrier functions that are used to satisfy the tasks
    """
    def __init__(self, tasks_list  : list[AlwaysTask|EventuallyTask], 
                       workspace   : Polyhedron, 
                       system      : ContinuousLinearSystem,
                       input_bound : Polyhedron ,
                       x_0         : np.ndarray,
                       minimize_robustness : bool = True,
                       k_gain      : float = -1.,
                       solver      : str   = "OSQP") -> None:
        
        """
        
        :param tasks_list: A list of tasks to be optimized.
        :type tasks_list: list[AlwaysTask|EventuallyTask]
        :param workspace: The workspace in which the tasks are to be optimized.
        :type workspace: Polyhedron
        :param system: The continuous linear system to be used for the optimization.
        :type system: ContinuousLinearSystem
        :param input_bound: The input bounds for the system.
        :type input_bound: Polyhedron
        :param x_0: The initial state of the system.
        :type x_0: np.ndarray
        :param minimize_robustness: Whether to minimize the robustness of the barrier functions.
        :type minimize_robustness: bool
        :param k_gain: The gain parameter for the optimization. If -1, it will be computed automatically.
        :type k_gain: float
        """
        
        self.tasks_list        : list[AlwaysTask|EventuallyTask] = tasks_list
        self.workspace         : Polyhedron                      = workspace
        self.system            : ContinuousLinearSystem          = system
        self.input_bounds      : Polyhedron                      = input_bound
        self.x_0               : np.ndarray                      = x_0
        self.minimize_r        : bool                            = minimize_robustness
        self.given_k_gain      :float                            = k_gain
        self.robustness        : float                           = 0. # initial guess for the robustness
        self.solver            : str                             = solver
        
        if solver == "OSQP":
            self.opti              : ca.Opti                         = ca.Opti("conic")
        elif solver == "ipopt":
            self.opti              : ca.Opti                         = ca.Opti()
        else :
            raise ValueError("The solver must be either 'OSQP' or 'ipopt'. Please choose one of the two solvers.")

        self.barriers                            : list[BarrierFunction]       = []
        self.time_varying_polyhedron_constraints : list[TimeVaryingConstraint] = []

        list_of_switches = sorted(list({ task.alpha_var_value for task in self.tasks_list if hasattr(task,"alpha_var")} | { task.beta_var_value for task in self.tasks_list} | {0.} ))
        
        additional_switches = []
        # # add point in between every pair of points
        # for jj in range(len(list_of_switches)-1):
        #     if list_of_switches[jj] != list_of_switches[jj+1]:
        #         additional_switches += [list_of_switches[jj] + (list_of_switches[jj+1] - list_of_switches[jj])/2]

        self.time_grid = sorted(list_of_switches+ additional_switches )
        
        if workspace.is_open:
            raise ValueError("The workspace is an open Polyhedron. Please provide a closed polyhedron")
        
        self.create_barriers()
        self.optimize_barriers()
        

    def create_barriers(self) :
        """
        From this function, the following tasks are accomplished:
            1) The given formula is sectioned sub tasks varphi.
            2) Each task var phi is converted into a barrier function.
        """

   
        for task in self.tasks_list:
            time_grid_barrier = [time for time in self.time_grid if  time <= task.beta_var_value]
            if hasattr(task,"alpha_var"):
                barrier = BarrierFunction(opti       = self.opti,
                                          polyhedron = task.polyhedron, 
                                          time_grid  = time_grid_barrier, 
                                          flat_time  = task.alpha_var_value)
            else:
                barrier = BarrierFunction(opti       = self.opti,
                                          polyhedron = task.polyhedron, 
                                          time_grid  = time_grid_barrier, 
                                          flat_time  = task.beta_var_value)

            self.barriers.append(barrier)

    
    def _make_high_order_corrections(self, system : ContinuousLinearSystem, k_gain : ca.MX) -> int:

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
        :type k_gain: ca.MX
        :return: The order of the barrier function.
        :rtype: int
        :raises ValueError: If the system is not controllable.
        """

        if not system.is_controllable():
            raise ValueError("The provided system is not controllable. As of now, only controllable systems can be considered for barriers optimization.")
        

        A = system.A 
        B = system.B
        I = np.eye(system.size_state)

        print("================================================")
        print("Correcting barriers for high order systems")
        print("================================================")
        for barrier in self.barriers :
            D = barrier.D
            c = barrier.c
            right_order  = False
            order = 0

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
                D_high_order = barrier.D@ca.mpower(A + I*k_gain,order) 
                c_high_order = ca.power(k_gain,order) * c
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
            raise ValueError("The input bounds are an open Polyhedron. Please provide a closed polyhedron")
        
        if self.input_bounds.num_dimensions != self.system.size_input:
            raise ValueError("The input bounds polyhedron must be in the same dimension as the system input. Given input bounds are in dimension " +
                             str(self.input_bounds.num_dimensions) + ", while the system input dimension is " + str(self.system.size_input))
        
        if not isinstance(self.system, ContinuousLinearSystem):
            raise ValueError("The system must be a ContinuousLinearSystem.")
        
        x_0 = self.x_0.flatten()
        if len(x_0) != self.workspace.num_dimensions:
            raise ValueError("The initial state must be a vector of the same dimension as the workspace.")

        
        vertices        = self.workspace.vertices.T # matrix [dim x,num_vertices]
        x_dim           = self.workspace.num_dimensions

        constraints = []

        # integrate the alpha and beta switching times into the time grid (so time steps between switching times will not be homogenous)
        k_gain     = self.opti.parameter() #! (when program fails try to increase control input and increase the k_gain. Carefully analyze the situation. Usually it should be at leat equal to 1.)
        order      = self._make_high_order_corrections(system = self.system, k_gain = k_gain)
        

        A = self.system.A
        B = self.system.B
        slack         = self.opti.variable()
        slack_penalty = 1E5
        self.opti.set_initial(slack,10)
        constraints  += [slack >= 0] # slack variable for the constraints

        # dynamic constraints
        for jj in range(len(self.time_grid)-1): # for each interval
            
            # Get two consecutive time intervals in the sequence.
            s_j        = self.time_grid[jj]
            s_j_plus_1 = self.time_grid[jj+1]-0.0001 # it is like the value at the limit

            # Create vertices set. 
            s_j_vec        = np.array([s_j for i in range(vertices.shape[1])]) # column vector
            s_j_plus_1_vec = np.array([s_j_plus_1 for i in range(vertices.shape[1])])
            V_j            = np.hstack((np.vstack((vertices,s_j_vec)) , np.vstack((vertices,s_j_plus_1_vec)))) # space time vertices
            U_j            = self.opti.variable(self.system.size_input, V_j.shape[1])                                         # spece of control input vertices
            
            # Input constraints.
            for kk in range(U_j.shape[1]):
                u_kk = U_j[:,kk]
                constraints += [ca.mtimes(self.input_bounds.A,u_kk) - self.input_bounds.b <= 0.] # input constraints

            
            # Forward-invariance constraints. 
            for active_task_index in self.active_barriers_map(s_j): # for each active task
                barrier = self.barriers[active_task_index] 

                e = barrier.e_vector(s_j) # time derivative of the gamma function for a given section
                c = barrier.c
                D = barrier.D
                
                for ii in range(V_j.shape[1]): # for each vertex
                    eta_ii   = V_j[:,ii]
                    u_ii     = U_j[:,ii]

                    eta_ii_not_time     = eta_ii[:-1]
                    time                = eta_ii[-1]
                    dyn                 = (ca.mtimes(A,eta_ii_not_time) + ca.mtimes(B,u_ii)) 
                    gama_at_time        = barrier.gamma_at_time(time) # gamma function for a given section

                    D_high_order        = barrier.D_high_order
                    c_high_order        = barrier.c_high_order
                    constraints        += [ca.mtimes(D_high_order,dyn) + e + k_gain * (ca.mtimes(D_high_order,eta_ii_not_time) + c_high_order + gama_at_time ) + slack>= 0.] # forward invariance constraints
                    

        # Add flatness constraint
        for barrier in self.barriers:
            constraints += barrier.get_constraints()

        # Inclusion constraints
        betas        = list({ barrier.time_grid[-1] for barrier in self.barriers} | {0.}) # set inside the list removes duplicates if any.
        betas        = sorted(betas)
        
        epsilon      = 1E-3
        zeta_vars    = self.opti.variable( x_dim, len(betas))
        # Impose zeta vars in the workspace
        for kk in range(1,zeta_vars.shape[1]):
            zeta_kk       =  zeta_vars[:,kk]
            constraints  += [ca.mtimes(self.workspace.A,zeta_kk) - self.workspace.b <= 0.] # inclusion constraints]
        
        for l in range(1,len(betas)):
            beta_l = betas[l]
            zeta_l = zeta_vars[:,l]

            for l_tilde in self.active_barriers_map(betas[l-1]) : # barriers active at lim t-> - beta_l is equal to the one active at time beta_{l-1}
                gamma_at_beta_l = self.barriers[l_tilde].gamma_at_time(beta_l-0.00001) # gamma function for a given section
                D               = self.barriers[l_tilde].D  
                c               = self.barriers[l_tilde].c

                constraints += [ca.mtimes(D,zeta_l) + c +  gamma_at_beta_l - epsilon >=  0.] # epsilon just to make sure the point is not at the boundary and it is strictly inside
        
        # set the zeta at beta=0 zero and conclude
        # initial state constraint
        
        zeta_0  = self.opti.variable(x_dim)
        for barrier in self.barriers:
            # at time beta=0 all tasks are active and they are in the first linear section of gamma
            
            c = barrier.c
            D = barrier.D
            gamma_at_zero = barrier.gamma_at_time(0.)

            constraints += [ca.mtimes(D,zeta_0) + c +  gamma_at_zero- epsilon >= 0.] # constraint at time 0 (epsilon just to be strictly inside)
        
        # initial state constraint
        constraints += [zeta_0 == x_0]

        # create problem and solve it
        cost = 0.

    
        if self.minimize_r:
            for barrier in self.barriers:
                # cost      +=  10000*cp.exp(-cp.sum(barrier.r_var)/10) # nonlinear cost
                cost      +=  -ca.sum(barrier.r_var)
            
            slack_cost = slack_penalty * slack 
            cost       = cost + slack_cost # add slack cost to the cost function

        else:
 
            slack_cost = slack_penalty * slack 
        
        self.opti.minimize(cost) # minimize the slack variable
        self.opti.subject_to(constraints) # add constraints to the optimization problem
        
        if self.solver == "OSQP":
            p_opts = {
                # QP options
                "error_on_fail": 0,
                "print_time": 0,
                "verbose": 1,
                "jit": True,  # Enable JIT compilation for performance
                "jit_options": {'compiler': 'ccache gcc',
                                'flags': ["-O2", "-pipe"]},
                'compiler': 'shell',
                "osqp" : 
                       {"eps_abs" : 1E-3, 
                        "eps_rel" : 1E-5, # maintain high relative accuracies 
                        "max_iter": 30000, 
                        "polish"  : True,
                        "verbose" : True,
                        "scaling"  : 40}, # important to maintain the polish to get the high accuracy solution for a linear program
            }
        

            self.opti.solver("osqp",p_opts)
        
        elif self.solver == "ipopt":
            p_opts = {
                # QP options
                "error_on_fail": 0,
                "print_time": 0,
                "verbose": 1,
                "hessian_approximation": "limited-memory", # use limited memory approximation for the Hessian
            #     "jit": True,  # Enable JIT compilation for performance
            #     "jit_options": {'compiler': 'ccache gcc',
            #                     'flags': ["-O2", "-pipe"]},
            #     'compiler': 'shell',
            }
        
            self.opti.solver("ipopt",p_opts)


        
        if self.given_k_gain < 0.:


            good_k_found = False

            print("Selecting a good gain k ...")
            # when barriers have order highr than 1, the problem is no more dpp and thus it takes a lot of time to solve it.
            if order > 1:
                k_vals = np.arange(0.000001, 0.5, 0.03)
            else :
                k_vals = np.arange(0.001, 1., 0.003)
            
            best_k    = k_vals[0]
            best_slak = 1E10
            
            # Parallelize using multiprocessing
            with tqdm(total=len(k_vals)) as pbar:
                for k_val in k_vals:
                    pbar.set_description(f"k = {k_val:.3f}")
                    self.opti.set_value(k_gain,k_val)
                    
                    try :
                        start =  perf_counter()
                        self.opti.solve()
                        end   = perf_counter()
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error in solving the problem with k_gain {k_val}. The error is the following {e}")
                        pbar.update(1)
                        continue
                    
                    
                    if self.opti.stats()["return_status"] == "solved" and self.opti.value(slack) < 1E-4:
                        best_k       = k_val
                        good_k_found = True
                        best_k_time  = end - start
            
                    elif self.opti.stats()["return_status"] == "solved" and self.opti.value(slack) > 1E-4 and not good_k_found:
                        if  self.opti.value(slack) <= best_slak :
                            best_slak   = self.opti.value(slack)
                            best_k      = k_val 
                            best_k_time = end - start
                    else :
                        continue
                

            if not good_k_found:
                print("No good k found. Please increase the range of k. Returing k with minimum violation")
            
            print("Best k found: ", best_k)
            print("Best k time: ", best_k_time)
            print("Best slack violation: ", best_slak)

            self.opti.set_value(k_gain, best_k)
            self.opti.solve()
                
        else:
            print("Given k_gain:",self.given_k_gain)

            self.opti.set_value(k_gain, self.given_k_gain)
            
            try :
                start =  perf_counter()
                self.opti.solve()
                end   = perf_counter()

                best_slak   = self.opti.value(slack)
                best_k      = self.given_k_gain
                best_k_time = end - start
                
            except Exception as e :
                print(f"Error in solving the problem with given k_gain {self.given_k_gain}. The error is the following")
                raise e

        print("===========================================================")
        print("Barrier functions optimization result")
        print("===========================================================")
        print("Status                         : ", "success" if self.opti.stats()["success"] else "failure")
        print("Solver time                    : ", best_k_time)
        print("Number of variables            : ", self.opti.debug.nx)
        print("Number of constraints          : ", self.opti.debug.ng)
        print("Optimal Cost (expluding slack) :" , self.opti.value(cost))
        print("Maximum Slack violation        : ", self.opti.value(slack))
        print('Robustness                     : ', min(ca.mmin(self.opti.value(barrier.r_var)) for barrier in self.barriers))
        print('K gain                         : ', self.opti.value(k_gain))
        print("-----------------------------------------------------------")
        print("Listing parameters per task")

        for task,barrier in zip(self.tasks_list,self.barriers) :
            print("Task Type       :", "Eventually" if isinstance(task,EventuallyTask) else "Always")
            print("Task Interval   :", task.time_interval)
            
            if isinstance(task,EventuallyTask):
                print("Task tau        :", task.beta_var_value)
            
            print("Barrier gamma_0 : ", self.opti.value(barrier.gamma_0_var))
            print("Barrier r       : ", self.opti.value(barrier.r_var))
            print("---------------------------------------------------")

        if not self.opti.stats()["success"] :
            print("Problem is not optimal. Terminate!")
            exit()
        
        # saving constraints as time_state constraints
        for barrier in self.barriers:
            D = barrier.D
            c = barrier.c

            for jj,time in enumerate(barrier.time_grid[:-1]) :
                
               
                e = self.opti.value(barrier.e_vector(time))
                g = self.opti.value(barrier.g_vector(time))

                start = barrier.time_grid[jj]
                end   = barrier.time_grid[jj+1]
                
                # convert from the form Dx + c >= 0 to Hx <= b

                H = - np.hstack((D,e[:,np.newaxis]))
                b =  c + g

                
                if np.abs((end-start))>= 1E-5: # avoid inserting the polyhedron if the interval of time is very small
                    self.time_varying_polyhedron_constraints += [TimeVaryingConstraint(start_time=start, end_time=end, H=H, b=b)] # the second part of the constraint it is just flat so we can remove it.
        
        self.robustness = min( ca.mmin(self.opti.value(barrier.r_var)) for barrier in self.barriers) # get the minimum robustness of the barriers
        
        return self.opti
    
    
    
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
            barrier.plot_gamma(ax,c=colors[jj], label = fr"$\gamma(t)_{jj}$")
            
        
        ax.axhline(0, color='black', linestyle='--', label = "Zero level")
        ax.set_title("Barrier functions")
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
            
            polyhedrons = []
            for constrain in self.time_varying_polyhedron_constraints:
                if t <= constrain.end_time and t >= constrain.start_time:
                    H = constrain.H[:,:-1]
                    b = constrain.b
                    e = constrain.H[:,-1]

                    polyhedron = Polyhedron(H, b - e*t)
                    polyhedrons.append(polyhedron)

            # create intersections
            if len(polyhedrons) > 0:
                intersection : Polyhedron = polyhedrons[0]
                for i in range(1, len(polyhedrons)):
                    intersection = intersection.intersect(polyhedrons[i])
                
                intersection.plot(ax,alpha=0.1, color = color)

            
    def get_barrier_as_time_varying_polyhedrons(self):
        return self.time_varying_polyhedron_constraints

    def save_polyhedrons(self, filename :str):
        """
        Save a list of polyhedrons (H, b) with intervals to a file.
        
        Args:
            polyhedrons_list: List of tuples like [(H1, b1, (min1, max1)), (H2, b2, (min2, max2)), ...]
            filename: Output file path (e.g., 'polyhedrons.json')
        """
        data = []
        for constraint in self.time_varying_polyhedron_constraints:
            data.append(constraint.as_dict())
        
        with open(filename, 'w') as f:
            json.dump(data, f)




def compute_polyhedral_constraints(formula       : Formula, 
                                    workspace    : Polyhedron, 
                                    system       : ContinuousLinearSystem, 
                                    input_bounds : Polyhedron, 
                                    x_0          : np.ndarray,
                                    plot_results : bool = False,
                                    k_gain       : float = -1.,
                                    polyhedron_file_name: str = "tvc_polyhedron.json") -> tuple[list["TimeVaryingConstraint"],float]:
    
    
    """
    :param formula: The formula to be optimized.
    :type formula: Formula
    :param workspace: The workspace polyhedron.
    :type workspace: Polyhedron
    :param system: The system to be used for the optimization.
    :type system: ContinuousLinearSystem
    :param input_bounds: The input bounds polyhedron.
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
                                          solver       = "OSQP")
    
    barrier_optimizer.save_polyhedrons(polyhedron_file_name)
    
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
        and dim is the state dimension. The constraint is expressed in space and time.
    """

    def __init__(self, start_time: float, end_time:float, H :np.ndarray, b:np.ndarray):
        
        
        self.start_time  :float       = start_time
        self.end_time    :float       = end_time
        self.H           : np.ndarray = H
        self.b           : np.ndarray = b

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
    
    def to_polyhedron(self) -> Polyhedron:
        """
        Convert the time-varying constraint to a Polytope object.
        
        Returns:
            Polytope object representing the time-varying constraint.
        """

        # add initial and final time constraints to the polyhedron
        row1 = np.hstack((np.zeros(self.H.shape[1]-1), np.array([1])))
        row2 = np.hstack((np.zeros(self.H.shape[1]-1), np.array([-1])))

        H_ext = np.vstack((self.H, row1, row2))
        b_ext = np.hstack((self.b, self.end_time, -self.start_time))

        # Plot the constraint as a polygon
        polyhedron = Polyhedron(H_ext, b_ext)


        return polyhedron
    
    def plot3d(self, ax: Optional[plt.Axes] = None,**kwords) -> None:
        """
        Plot the time-varying constraint.
        
        Args:
            ax: Matplotlib Axes object to plot on. If None, create a new figure and axis.
        """
        

        polyhedron = self.to_polyhedron()
        num_dims = polyhedron.num_dimensions

        if num_dims == 3: # 2d polyhedron + time
        
            if ax is None:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

            polyhedron.plot(ax, **kwords)
            
            # Set title and labels
            ax.set_title(f'Time-Varying Constraint [{self.start_time}, {self.end_time}]')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        elif num_dims >= 3: # 3d polyhedron + time
            if ax is None:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

            for t in np.linspace(self.start_time, self.end_time, 10):
                e = self.H[:,-1]
                b = self.b
                H = self.H[:,:3] # plot first three dimensions
                b_t = b - e*t
                polyhedron = Polyhedron(H, b_t)
                # Plot the constraint as a polygon
                polyhedron.plot(ax,**kwords)

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
        

        polyhedron = self.to_polyhedron()
        num_dims = polyhedron.num_dimensions

        if num_dims != 3:
            raise ValueError("Time-varying constraints can only be plotted in 2D.")
        
        if ax is None:
            fig,ax = plt.subplots(figsize=(10, 4))
        
        for t in np.linspace(self.start_time, self.end_time, 10):
            e = self.H[:,-1]
            b = self.b
            H = self.H[:,:-1]
            b_t = b - e*t
            polyhedron = Polyhedron(H, b_t)
            # Plot the constraint as a polygon
            polyhedron.plot(ax)

        

        # Set title and labels
        ax.set_title(f'Time-Varying Constraint [{self.start_time}, {self.end_time}]')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        