import numpy as np
import cvxpy as cp

from   typing     import Optional, Union
from   matplotlib import pyplot as plt
from   tqdm       import tqdm
import json
from multiprocessing import Pool


from .logic         import (Formula, 
                           AndOperator,    
                           get_fomula_type_and_predicate_node, 
                           OrOperator,
                           UOp, FOp, GOp, 
                           Predicate, PredicateNode)





from .utils         import TimeInterval
from .linear_system import ContinuousLinearSystem

from ..polytope  import Polytope


class BarrierFunction :
    """

    This is an internal class that represents a general time-varying constraint of the form :math:`Dx + c+ \gamma(t) >=0`.
    
    
    We represent such constraint as a barrier function with multiple output as :math:`b(x,t) = Dx +c + \gamma(t)`, where 
    :math:`Dx + c >=0` represents a polytope. 

    The function gamma is a piexe wise linear function of the form :math:`\gamma(t) = e \cdot t + g`.
    """
    def __init__(self, polytope: Polytope ) -> None:
        """
        Initialize the barrier function with a given polytope.

        :param polytope: Polytope object representing the constraint.
        :type polytope: Polytope
        """
        
        self.polytope   = polytope
        self.task_type  = None                              # defines the type of tasks related to this barrier function
        self.interval_satisfaction : TimeInterval = None    # defines the interval of uncertainity in which the task has to be

        self.alpha_var   : cp.Variable =  cp.Variable(nonneg=True)
        self.beta_var    : cp.Variable =  cp.Variable(nonneg=True)
        self.gamma_0_var : cp.Variable =  cp.Variable((self.polytope.num_hyperplanes),nonneg=True)
        self.r_var       : cp.Variable =  cp.Variable(pos=True)

        self._D_high_order = None
        self._c_high_order = None
        
        #! do not touch!
        self._a_prime : float = 0.0
        self._b_prime : float = 0.0
        
        
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

    @property
    def e1_var(self):
        e_vec = - (self.gamma_0_var/self.alpha) 
        return e_vec
    @property
    def e2_var(self):
        e_vec = np.zeros(self.polytope.num_hyperplanes)
        return e_vec
    @property
    def g1_var(self) :
        g_vec = (self.gamma_0_var  - self.r_var)
        return g_vec
    @property
    def g2_var(self) :
        g_vec = -self.r_var *np.ones(self.polytope.num_hyperplanes) 
        return g_vec
    @property
    def e1_value(self):
        return - self.gamma_0/self.alpha 
    
    @property
    def e2_value(self):
        return np.zeros(self.polytope.num_hyperplanes)
    
    @property
    def g1_value(self):
        return (self.gamma_0 - self.r) 
    
    @property
    def g2_value(self):
        return - self.r * np.ones(self.polytope.num_hyperplanes)
    

    def upsilon(self,t:float)-> int :
        """
        Return 1 or 2 depending on which linear section of the function the given time is in.

        if t < alpha then return 1\n
        if t > alpha and t < beta then return 2\n
        if t > beta then raise ValueError\n

        :param t: Time value.
        :type t: float
        :return: 1 or 2 depending on the section.
        :rtype: int


        """
        t = float(t)

        if (t >= 0.) and (t < self.alpha):
            return 1
        elif (t >= self.alpha) and (t <= self.beta):
            return 2
        else :
            raise ValueError("The given time time is outside the range [0,beta].")
        
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


def solve_with_k(k_val, constraints, cost, k_gain):
    # Rebuild the problem each time for true parallelism
    problem = cp.Problem(cp.Minimize(cost), constraints)
    k_gain.value = k_val
    try:
        problem.solve(warm_start=True)
        if problem.status == cp.OPTIMAL:
            return k_val
    except:
        pass
    return None


class TasksOptimizer:
    
    def __init__(self,formula : Formula, workspace: Polytope, system : ContinuousLinearSystem ) -> None:
        
        self.formula           : Formula                 = formula
        self._workspace        : Polytope                = workspace
        self.system            : ContinuousLinearSystem  = system

        self._varphi_formulas  : list[Formula]          = [] # subformulas G,F,FG,GF
        self._barriers         : list[BarrierFunction]  = []
        self._time_constraints : list[cp.Constraint]    = []

        self.task_durations : list[dict]              = []

        self._time_varying_polytope_constraints : list[TimeVaryingConstraint] = []

        if workspace.is_open:
            raise ValueError("The workspace is an open Polyhedron. Please provide a closed polytope")

    def _create_barriers_and_time_constraints(self) :
        """
        From this function, the following tasks are accomplished:
            1) The given formula is sectioned sub tasks varphi.
            2) Each task var phi is converted into a barrier function.
        """

     
        barriers         : list[BarrierFunction]    = []
        time_constraints : list[cp.Constraint]      = []

        is_a_conjunction = False

        # Check that the provided formula is within the fragment of allowable formulas.
        if isinstance(self.formula.root,OrOperator): #1) the root operator can be an or, but it not implemented for now.
            raise NotImplementedError("OrOperator is not implemented yet. Wait for it. it is coming soon.")
        elif isinstance(self.formula.root,AndOperator): # then it is a conjunction of single formulas
            is_a_conjunction = True
        else: #otherwise it is a single formula
            pass
        
        # subdivide in sumbformulas
        possible_fomulas = ["G", "F", "FG", "GF"]
        if is_a_conjunction :
            for child_node in self.formula.root.children : # take all the children nodes and check that the remaining formulas are in the predicate
                varphi = Formula(root = child_node)
                varphi_type, predicate_node = get_fomula_type_and_predicate_node(formula = varphi)

                if varphi_type in ["G","F","FG"] :
                    self._varphi_formulas += [varphi]
                    
                elif varphi_type == "GF" : # decompose
                    g_interval : TimeInterval = child_node.interval
                    f_interval : TimeInterval = child_node.children[0].interval # the single children of the always operator is the eventually

                    nf_min = np.ceil((g_interval.b - g_interval.a)/(f_interval.b - f_interval.a)) # minimum frequency of repetition
                    m      = g_interval.a + f_interval.a

                    interval       = g_interval.b - g_interval.a
                    interval_prime = f_interval.b - f_interval.a
                    delta_bar   = 1/nf_min *(interval/interval_prime)

                    a_bar_w = m

                    for w in range(1,int(nf_min)+1):
                        
                        # keep order of defitiion as a_w_bar gets redefined
                        b_bar_w = a_bar_w + w* 1* interval_prime
                        a_bar_w = a_bar_w + w* 1* interval_prime

                        self._varphi_formulas += [ FOp(a = a_bar_w,b = b_bar_w) >> Predicate(polytope= predicate_node.polytope,dims = predicate_node.dims, name = predicate_node.name) ]
                else:
                    raise ValueError("At least one of the formulas set in conjunction is not within the currently allowed grammar. Please verify the predicates in the formula.")

        
        else: # if the task is not a conjunction then it is already an elementary task varphi
            varphi = self.formula
            varphi_type, predicate_node = get_fomula_type_and_predicate_node(formula = self.formula)
            
            if varphi_type == "GF" : # decompose
                g_interval : TimeInterval = varphi.root.interval
                f_interval : TimeInterval = varphi.root.children[0].interval

                nf_min = np.ceil((g_interval.b - g_interval.a)/(f_interval.b - f_interval.a)) # minimum frequency of repetition
                m      = g_interval.a + f_interval.a

                interval       = g_interval.b - g_interval.a
                interval_prime = f_interval.b - f_interval.a
                delta_bar   = 1/nf_min *(interval/interval_prime)

                a_bar_w = m

                for w in range(1,int(nf_min)+1):
                    
                    # keep order of defitiion as a_w_bar gets redefined
                    b_bar_w = a_bar_w + w* 1* interval_prime
                    a_bar_w = a_bar_w + w* 1* interval_prime
                    self._varphi_formulas += [ FOp(a = a_bar_w,b = b_bar_w) >> Predicate(polytope= predicate_node.polytope,dims = predicate_node.dims, name = predicate_node.name) ]
            
            elif varphi_type in possible_fomulas :
                self._varphi_formulas += [varphi]
            else:
                raise ValueError("The given formula is not in the allowed fragment")

         
        
        print("============================================================")
        print("Enumerating tasks")
        print("============================================================")
        
        for varphi in self._varphi_formulas:
            try :
                varphi_type, predicate_node = get_fomula_type_and_predicate_node(formula = varphi)
            except ValueError:
                raise ValueError("At least one of the formulas set in conjunction is not within the currently allowed grammar. Please verify the predicates in the formula.")
            
            root          : Union[FOp,UOp,GOp] = varphi.root
            time_interval : TimeInterval       = root.interval
            dims          : list[int]          = predicate_node.dims
            polytope      : Polytope           = predicate_node.polytope

            try : 
                C   = self.system.output_matrix_from_dimension(dims) 
            except Exception as e:
                print(f"Error in output matrix creation. The error stems from the fact that the required dimension of one or more predicates in the formulas" +
                      f"is out of range for the system e.g. a predicate is enforcing a specification of the state with index 3 but the state dimension is only 2. " +
                      f"The raised exception is the following")
                raise e


            # Change polytope to accomodate the dimension of the system using the output matrix
            A        = polytope.A@C # Expansion of the polytope to the dimension of the system.
            b        = polytope.b
            polytope = Polytope(A,b)
            
            # Create barrier functions for each task
            if varphi_type == "G" :

                # Create barrier function.
                barrier : BarrierFunction  = BarrierFunction(polytope)
                barrier.task_type          = varphi_type
                
                ## Give initial guess to the solver
                barrier.alpha_var.value = time_interval.a
                barrier.beta_var.value  = time_interval.b

                # save interval uncertainity for conflicting conjunction detection
                barrier.interval_satisfaction = time_interval
                
                # add barrier to the list
                barriers.append(barrier)
                
                # create time constraints
                time_constraints += [barrier.alpha_var == time_interval.a, barrier.beta_var == time_interval.b]
                
                # add tasks to the list for plotting
                start_time = time_interval.a
                duration   =  time_interval.b - start_time
                self.task_durations.append({'start_time': start_time, 'duration': duration, 'type': varphi_type})

            elif varphi_type == "F" :

                # Create barrier function.
                barrier : BarrierFunction  = BarrierFunction(polytope)
                barrier.task_type          = varphi_type
                
                ## Give initial guess to the solver
                barrier.alpha_var.value = time_interval.a
                barrier.beta_var.value  = time_interval.b

                # save interval uncertainity for conflicting conjunction detection
                barrier.interval_satisfaction = time_interval
                
                # add barrier to the list
                barriers.append(barrier)
                
                # create time constraints
                time_constraints += [barrier.alpha_var >= time_interval.a, 
                                    barrier.beta_var   == barrier.alpha_var,
                                    barrier.beta_var   <= time_interval.b]
                
                # add tasks to the list for plotting
                start_time = time_interval.a
                duration   =  (time_interval.b - start_time) + 0.1 # just for plotting added a o.1 to make singleton intervals visible
                self.task_durations.append({'start_time': start_time, 'duration': duration, 'type': varphi_type})
  

            elif varphi_type == "FG":


                time_interval_prime : TimeInterval    = root.children[0].interval # extract interval of the operator G

                # Create barrier function.
                barrier : BarrierFunction  = BarrierFunction(polytope)
                barrier.task_type          = varphi_type
                
                ## Give initial guess to the solver
                barrier.alpha_var.value = time_interval.get_sample() + time_interval_prime.a
                barrier.beta_var.value  = barrier.alpha_var.value + (time_interval_prime.b - time_interval_prime.a)

                # save interval uncertainity for conflicting conjunction detection
                barrier.interval_satisfaction = time_interval
                
                # add barrier to the list
                barriers.append(barrier)
                
                # create time constraints
                time_constraints += [barrier.alpha_var >= time_interval.a +  time_interval_prime.a, 
                                     barrier.alpha_var <= time_interval.b +  time_interval_prime.a,
                                     barrier.beta_var  == barrier.alpha_var + (time_interval_prime.b - time_interval_prime.a)]
                
    
                start_time = time_interval.a + time_interval_prime.a
                duration   =  (time_interval.b  + time_interval_prime.b) - start_time
                self.task_durations.append({'start_time': start_time, 'duration': duration, 'type': varphi_type})    
                
                # these are neeeded to detect a possible conflicting conjuntion
                barrier._a_prime = time_interval_prime.a
                barrier._b_prime =time_interval_prime.b

            else :
                pass # formulas of type GF have already been decomposed into F formulas
            
            print("Found Tasks of type: ", varphi_type)
            print("Start time: ", start_time)
            print("Duration: ", duration)
            print("---------------------------------------------")


        print("====== Enumeration completed =============================================")

        self._barriers         = barriers
        self._time_constraints = time_constraints
        print("========= Cheking for possible conflicting conjuncitons ==================")
        self.detect_conflicting_conjunctions_and_give_good_time_guesses()

    
    
    def distace_between_polytopes(self,barrier_1 : BarrierFunction, barrier_2: BarrierFunction):
        """
        Use cvxpy to computer the distance between two polytopes inside two barrier functions"
        """

        x = cp.Variable((self.system.size_state,1))
        y = cp.Variable((self.system.size_state,1))

        A1,b1 = barrier_1.polytope.A, barrier_1.polytope.b
        A2,b2 = barrier_2.polytope.A, barrier_2.polytope.b

        constraints = [A1@x <= b1, A2@y <= b2]
        cost = cp.norm(x-y,2)
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        return cost.value
    
    
    def detect_conflicting_conjunctions_and_give_good_time_guesses(self)-> None :
        """ This check can be expensive if you have many tasks. But it can be useful to avoid problems"""
        
        always_barriers = [barrier for barrier in self._barriers if barrier.task_type == "G"]
        
        for barrier in self._barriers:
            
            time_interval = barrier.interval_satisfaction
            if barrier.task_type == "F":
                for always_barrier in always_barriers:
                    always_time_interval = always_barrier.interval_satisfaction
                    if time_interval in always_time_interval :
                        if self.distace_between_polytopes(barrier,always_barrier) < 1E-2: # two separate if to avoid computing the distance condition for nothing
                            message = (f"Found conflicting conjunctions: A task of type F_[a,b]\mu is conflicting with a task of type G_[a',b']\mu" +
                                    "The time interval of the eventually is fully contained the the time interval of the always and the predicates have no intersection")
                            raise Exception(message)
                    else : # give good initial guess for the eventually to have both alpha and beta outside the always interval (on the right)
                        
                        epsilon = 1E-3
                        barrier.alpha_var.value =  always_time_interval.b + epsilon
                        barrier.beta_var.value  =  always_time_interval.b + epsilon

            elif barrier.task_type == "G":
                for always_barrier in always_barriers:
                    if barrier != always_barrier:
                        always_time_interval = always_barrier.interval_satisfaction
                        if (time_interval / always_time_interval) is not None:
                            if self.distace_between_polytopes(barrier,always_barrier) < 1E-2: # two separate if to avoid computing the distance condition for nothing
                                message = (f"Found conflicting conjunctions: A task of type G_[a,b]\mu is conflicting with a task of type G_[a',b']\mu" +
                                        "The interval of the tasks is intersectinig ")
                                raise Exception(message)
            
            elif barrier.task_type == "FG":
                
                a_prime = barrier._a_prime
                b_prime = barrier._b_prime

                for always_barrier in always_barriers:
                        always_time_interval = always_barrier.interval_satisfaction

                        task_always_starts_in_the_always = (a_prime+time_interval) / always_time_interval is not None
                        task_always_ends_in_the_always   = (b_prime+time_interval) / always_time_interval is not None
                        
                        if task_always_starts_in_the_always or task_always_ends_in_the_always:
                            if self.distace_between_polytopes(barrier,always_barrier) < 1E-2: # two separate if to avoid computing the distance condition for nothing
                                message = (f"Found conflicting conjunctions: A task of type F_[a,b]G_[a',b']\mu is conflicting with a task of type G_[a_bar,b_bar]\mu" +
                                           "The time interval of the FG formula is such that the task either always starts in the interval of the always or it always ends in the interval of the always but the two do not have an intersecing predicate")
                                raise Exception(message)
                        else : # give good initial guess for the eventually to have both alpha and beta outside the always interval (on the right)
                            
                            epsilon = 1E-3
                            barrier.alpha_var.value =  always_time_interval.b + a_prime
                            barrier.beta_var.value  =  barrier.alpha_var.value + (b_prime - a_prime)


    
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
        I = np.eye(system.size_state)

        print("================================================")
        print("Correcting barriers for high order systems")
        print("================================================")
        for barrier in self._barriers :
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
                D_high_order = barrier.D@cp.power(A + I*k_gain,order) 
                c_high_order = cp.power(k_gain,order) * c
                barrier.set_high_order_constraints(D_high_order, c_high_order)
            
        return order

    def make_time_schedule(self) :
        """
        From this function, the following tasks are accomplished:
            1) The given formula is sectioned sub tasks varphi.
            2) Each task var phi is converted into a barrier function.
            3) A time schedule for each task by assigning proper satisfaction and conclusion times to each task based on the temporal operators of the task.
        """

        self._create_barriers_and_time_constraints()
        
        # create optimization problem 

        cost = 0
        normalizer = 50.
        lift_up_factor = 10.
        for barrier_i in self._barriers:
            for barrier_j in self._barriers :
                if barrier_i != barrier_j and barrier_i.task_type != "G":

                    # give cost based on the guessed ordering of the tasks: we want to maximazie distance between each point.
                    # In order to do so we
                    if barrier_i.alpha_var.value >= barrier_j.alpha_var.value :
                        cost += lift_up_factor*cp.exp(-(barrier_i.alpha_var - barrier_j.alpha_var)/normalizer)
                    else:
                        cost += lift_up_factor*cp.exp(-(barrier_j.alpha_var - barrier_i.alpha_var)/normalizer)
                    
                    if barrier_i.beta_var.value >= barrier_j.alpha_var.value :
                        cost += lift_up_factor*cp.exp(-(barrier_i.beta_var - barrier_j.alpha_var)/normalizer)
                    else:
                        cost += lift_up_factor*cp.exp(-(barrier_j.alpha_var - barrier_i.beta_var)/normalizer)

                    if barrier_i.alpha_var.value >= barrier_j.beta_var.value :
                        cost += lift_up_factor*cp.exp(-(barrier_i.alpha_var - barrier_j.beta_var)/normalizer)
                    else:
                        cost += lift_up_factor*cp.exp(-(barrier_j.beta_var - barrier_i.alpha_var)/normalizer)

                    if barrier_i.beta_var.value >= barrier_j.beta_var.value :
                        cost += lift_up_factor*cp.exp(-(barrier_i.beta_var - barrier_j.beta_var)/normalizer)
                    else:
                        cost += lift_up_factor*cp.exp(-(barrier_j.beta_var - barrier_i.beta_var)/normalizer)


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
        for i, task in enumerate(self.task_durations):
            ax.broken_barh([(task["start_time"], task["duration"])], (i - 0.4, 0.8), facecolors='tab:blue')

        # Labeling
        ax.set_xlabel('Time')
        ax.set_ylabel("tasks")
        ax.set_yticks(range(len(self.task_durations)))
        ax.set_yticklabels([rf'\phi =  {task["type"]}  \mu' for task in self.task_durations])
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

    
    def optimize_barriers(self, input_bounds: Polytope , x_0 : np.ndarray) :
        
        
        if input_bounds.is_open:
            raise ValueError("The input bounds are an open Polyhedron. Please provide a closed polytope")
        
        if input_bounds.num_dimensions != self.system.size_input:
            raise ValueError("The input bounds polytope must be in the same dimension as the system input. Given input bounds are in dimension " +
                             str(input_bounds.num_dimensions) + ", while the system input dimension is " + str(self.system.size_input))
        
        if not isinstance(self.system, ContinuousLinearSystem):
            raise ValueError("The system must be a ContinuousLinearSystem.")
        
        x_0 = x_0.flatten()
        if len(x_0) != self._workspace.num_dimensions:
            raise ValueError("The initial state must be a vector of the same dimension as the workspace.")

        
        vertices              = self._workspace.vertices.T # matrix [dim x,num_vertices]
        x_dim                 = self._workspace.num_dimensions
        set_of_time_intervals = list({ barrier.alpha for barrier in self._barriers} | { barrier.beta for barrier in self._barriers} | {0.}) # repeated time instants will be counted once in this way
        ordered_sequence      = sorted(set_of_time_intervals)
        k_gain                = cp.Parameter(pos=True) #! (when program fails try to increase control input and increase the k_gain. Carefully analyze the situation. Usually it should be at leat equal to 1.)
        
        order = self._make_high_order_corrections(system = self.system, k_gain = k_gain)
        

        A = self.system.A
        B = self.system.B
        slack         = cp.Variable(nonneg = True)
        slack_penalty = 10000 

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
            U_j            = cp.Variable((self.system.size_input, V_j.shape[1]))                                         # spece of control input vertices
            
            # Input constraints.
            for kk in range(U_j.shape[1]):
                u_kk = U_j[:,kk]
                constraints += [input_bounds.A @ u_kk <= input_bounds.b]
            # Forward-invariance constraints. 
            for active_task_index in self.active_barriers_map(s_j): # for each active task
                barrier = self._barriers[active_task_index] 
                
                # select correct section (equivalent to upsilon)
                if barrier.upsilon(s_j) == 1 : # then (s_i,s_i_plus_1) in [0,\alpha_l]
                    e = barrier.e1_var
                    g = barrier.g1_var
                else: # then (s_i,s_i_plus_1) in [\alpha_l,\beta_l]
                    e = barrier.e2_var
                    g = barrier.g2_var

                c = barrier.c
                D = barrier.D
                
                for ii in range(V_j.shape[1]): # for each vertex
                    eta_ii   = V_j[:,ii]
                    u_ii     = U_j[:,ii]

                    eta_ii_not_time     = eta_ii[:-1]
                    time                = eta_ii[-1]

                    dyn                 = (A @ eta_ii_not_time + B @ u_ii) 

                    D_high_order = barrier.D_high_order
                    c_high_order = barrier.c_high_order

                    constraints        += [D_high_order @ dyn + e + k_gain * (D_high_order @ eta_ii_not_time + e*time + c_high_order + g ) + slack>= 0]

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
                    e = self._barriers[l_tilde].e1_var
                    g = self._barriers[l_tilde].g1_var
                else:
                    e = self._barriers[l_tilde].e2_var
                    g = self._barriers[l_tilde].g2_var

                D = self._barriers[l_tilde].D  
                c = self._barriers[l_tilde].c
                constraints += [D @ zeta_l + e * beta_l + c + g >= 0]
        
        # set the zeta at beta=0 zero and conclude
        # initial state constraint
        zeta_0  = zeta_vars[:,0]
        epsilon = 1E-1 # just to be strictly inside
        for barrier in self._barriers:
            # at time beta=0 all tasks are active and they are in the first linear section of gamma
            e = barrier.e1_var
            g = barrier.g1_var
            
            c = barrier.c
            D = barrier.D

            constraints += [D @ zeta_0 + c + g >= epsilon]
        
        # initial state constraint
        constraints += [zeta_0 == x_0]

        # create problem and solve it
        cost = 0
        for barrier in self._barriers:
            cost += -barrier.r_var
            
        cost += slack_penalty * slack
        problem = cp.Problem(cp.Minimize(cost), constraints)
        good_k_found = False


        print("Selcting a good gain k ...")
        # when barriers have order highr than 1, the problem is no more dpp and thus it takes a lot of time to solve it.
        if order >= 1:
            k_vals = np.arange(0.01, 0.06, 0.05)
        else :
            k_vals = np.arange(0.01, 0.06, 0.05)
        
        best_k    = k_vals[0]
        best_slak = 1E10
        # Parallelize using multiprocessing
        for k_val in tqdm(k_vals):
            k_gain.value = k_val
    
            problem.solve(warm_start=True, verbose=False,solver="MOSEK")
            if problem.status == cp.OPTIMAL and slack.value < 1E-5:
                best_k = k_val
                good_k_found = True
                break
            else:
                if slack.value <= best_slak :
                    best_slak = slack.value
                    best_k    = k_val 
            print("-----------------------------------------------------------")
            print("K value : ", k_val)
            print("Slack value : ", slack.value)

        if not good_k_found:
            print("No good k found. Please increase the range of k. Returing k with minimum violation")
            k_gain.value = k_val
            problem.solve(warm_start=True, verbose=False,solver="MOSEK")




        print("===========================================================")
        print("Barrier functions optimization result")
        print("===========================================================")
        print("Status        : ", problem.status)
        print("Optimal value : ", np.sum([-barrier.r_var.value for barrier in self._barriers]))
        print("Maximum Slack violation : ", slack.value)
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
            D = barrier.D
            c = barrier.c
            e1 = barrier.e1_value
            e2 = barrier.e2_value
            g1 = barrier.g1_value
            g2 = barrier.g2_value

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
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Transforms into 3D

        self._workspace.plot(ax, alpha=0.01)
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
                intersection : Polytope = polytopes[0]
                for i in range(1, len(polytopes)):
                    intersection = intersection.intersect(polytopes[i])
                
                intersection.plot(ax,alpha=0.1)
                # ax.set_title(f"Time-varying level set at t={t:.2f}")
        

            


    
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
        