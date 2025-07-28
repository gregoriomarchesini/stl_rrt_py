
import numpy as np
import matplotlib.pyplot as plt
from   typing                     import TypedDict
from   tqdm                       import tqdm
from   matplotlib.collections     import LineCollection
from   mpl_toolkits.mplot3d.art3d import Line3DCollection
from   scipy.spatial              import KDTree
import time
import casadi as ca

from stl_tool.environment.map         import Map
from stl_tool.polyhedron              import Polyhedron,selection_matrix_from_dims
from stl_tool.stl.parameter_optimizer import TimeVaryingConstraint

from stl_tool.planners.samplers import BiasedSampler, UnbiasedSampler
from stl_tool.splines.bezier    import PBezierCurve, BezierCurve

BIG_NUMBER = 1e10
     
class RRTSolution(TypedDict):
    path_trj     : list[np.ndarray]
    cost         : float
    iter         : int
    nodes        : list[np.ndarray]
    clock_time   : float



class SplineConnector:
    """
    C1 spline connector class.
    """

    def __init__(self, dim : int , stl_constraints  : list[TimeVaryingConstraint] = [], order : int = 3) -> None:
        
        """
        :dim: Dimension of the system.
        :type dim: int
        :param stl_constraints: List of time-varying constraints for the STL specifications. The constraints are of the form H[x,t] <= b and are imposed on the control points of the b-spline.
        :type stl_constraints: list[TimeVaryingConstraint]
        """
        
        self.stl_constraints    : list[TimeVaryingConstraint] = stl_constraints
        self.dim                : int = dim
        self.order              : int = order

        
        # check that the dimension of the constraint is correct
        if len(self.stl_constraints):

            if not all(self.dim == constraint.system_dim for constraint in self.stl_constraints):
                raise ValueError(f"Give system dimension is {self.dim} but some/all constraints don't comply with this dimension.")


        self.opti         = ca.Opti("conic")  # Create an Opti instance for optimization
        self.x_curve      = PBezierCurve(order = self.order, dim   = self.dim, opti = self.opti) # create a third order bezier curve for the position.
        self.x_dot_curve  = self.x_curve.get_derivative()  # velocity curve is the derivative of the position curve

        
        self.x0_par = self.opti.parameter(self.dim)  # initial position parameter
        self.x1_par = self.opti.parameter(self.dim)
        self.x1_dot_par = self.opti.parameter(self.dim)  # initial velocity parameter
        self.x0_dot_par = self.opti.parameter(self.dim)
        
        self.t0_par = self.opti.parameter(1)          # initial time parameter
        self.t1_par = self.opti.parameter(1)          # final time parameter

        self.activation_parameters = self.opti.parameter(len(self.stl_constraints))  # activation parameters for the constraints
        
        
        self._setup()


    
    def _setup(self) :


        
        constraints = []
        # constraints for the initial and final velocities 
        constraints.append(self.x0_par == self.x_curve.evaluate(0))  # initial position
        constraints.append(self.x1_par == self.x_curve.evaluate(1))

    
        delta_t = (self.t1_par - self.t0_par)/self.order # delta time between two time points
        
          
        time_points = [self.t0_par + i*delta_t for i in range(self.order+1)]  # time points for the cubic spline curve
        
        timed_control_points = []
        for point,t in zip(self.x_curve.control_points,time_points):
            timed_control_points.append(ca.horzcat(point, t)) # (p_0,t_0), (p_1,t_1), ... , (p_n,t_n)

        
        for timed_control_point in timed_control_points:   
            # state time constraints (Only difference with standard set point tracking MPC)
            for jj,constraint in enumerate(self.stl_constraints):
                H,b              = constraint.to_polytope()
                activation_bit   = self.activation_parameters[jj]
                constraints += [ca.mtimes(H,timed_control_point) <= b + (1 - activation_bit) * 1e6] # add time varying constraints

       

        # jerk cost along the lines
        cost   = 0
        t_span = np.linspace(0, 1, 1000)
        dt     = t_span[1] - t_span[0]  # Time step for numerical integration
        
        for t in t_span:
            jerk_value = self.x_dot_curve.evaluate(t)
            cost += ca.sumsqr(jerk_value)*dt

        self.opti.subject_to(constraints)
        self.opti.minimize(cost)
        
        self.opti.solver("osqp")  # Set the solver for the optimization problem

        self.planner = self.opti.to_function('MPCPlanner', [self.x0_par,     # give initial state 
                                                            self.x1_par,     # give final state
                                                            self.t0_par,     # initial time
                                                            self.t1_par,     # final time
                                                            self.activation_parameters],    # set of active constraints
                                                            
                                                            [self.x_curve.stacked_control_points, cost], ) # get control points of the spline as output
        
        
    
    def get_active_constraints_vector(self,t0: float, t1:float) -> np.ndarray :

        active_constraints = np.zeros(len(self.stl_constraints))
        for jj,constraint in enumerate(self.stl_constraints):
            t0_in_interval = (constraint.start_time <= t0 <= constraint.end_time)
            t1_in_interval = (constraint.start_time <= t1 <= constraint.end_time)

            if (t0_in_interval and not t1_in_interval) or (not t0_in_interval and t1_in_interval):
                raise ValueError(f"time t0 and t1 must be both contaioned or both not contained in the time interval of each STL constraint,"
                                 f" But there is one constraint such that t0: {t0}, t1: {t1}, and constraint bounds are : {constraint.start_time}, {constraint.end_time}.") 
            elif t0_in_interval and t1_in_interval:
                active_constraints[jj] = 1.0

        
        return active_constraints

    
    def connect(self, x0 : np.ndarray, x1 : np.ndarray , t0 : float, t1 : float) -> BezierCurve:

        active_constraints =  self.get_active_constraints_vector(t0, t1)  # get the active constraints vector
        
        # set the parameters 
        control_points_array, cost    = self.planner(x0, x1, t0, t1,active_constraints)  # Call the planner function with the parameters
        optimal_control_points  = [control_points_array[i*self.dim:(i+1)*self.dim].full().flatten() for i in range(self.position_curve.n)]  # Convert the output to a list of numpy arrays

        x_curve_opt     = BezierCurve(optimal_control_points)

        return x_curve_opt
        
    

class StlRRTStarSpline :
    """
    RRT* planner for STL specifications with spline trajectories.
    
    """
    def __init__(self, start_state      : np.ndarray, 
                       stl_constraints  : list[TimeVaryingConstraint],
                       map              : Map,
                       order            : int   = 3,
                       max_iter         : int   = 100000, 
                       space_step_size  : float = 0.3, 
                       time_step_size   : float = 0.1,
                       bias_future_time : bool  = True,
                       verbose          : bool  = False ,
                       rewiring_radius  : float = -1,
                       rewiring_ratio   : int   = 2,
                       biasing_ratio    : int   = 2) -> None :
        
        
        start_state            = np.array(start_state).flatten()
        self.start_time        = 0.                                                        # initial time
        self.start_cost        = 0.                                                        # start cost of the initial node
        self.start_node        = np.array([*start_state, self.start_time])                 # Start node (node = (state,time))
        self.map               = map                                                       # map containing the obstacles
        self.max_iter          = max_iter                                                  # maximum number of iterations
        self.space_step_size   = space_step_size                                           # mximum distance of the new sampled node from the current tree
        self.time_step_size    = time_step_size                                            # maximum time step for the new sampled node 
        self.space_time_dist   = np.sqrt(self.space_step_size**2 + self.time_step_size**2) # maximum space-time distance of the new node
        self.stl_constraints   = stl_constraints                                           # stl constraints
        self.spline_order      = order                                                     # order of the spline
        self.switch_times      = list(set([constraint.start_time for constraint in stl_constraints] + [constraint.end_time for constraint in stl_constraints]))  # times at which the constraints switch
        
        if order < 3:
            raise ValueError("Spline order must be at least 3 for C1 continuity.")
        
        
        if not start_state in self.map.workspace:
            raise ValueError(f"Start state {start_state} is not in the workspace.")
        
                  
        # Create RRT list
        self.tree              : list[np.ndarray]  = [self.start_node]       # list of nodes in the RRT tree.
        self.current_best_cost : float             = BIG_NUMBER              # current best cost.
        self.cost              : list[float]       = [0.]                    # cost for each node.
        self.trajectories      : list[BezierCurve] = [BezierCurve()]                      # for each node we stroe the trajectory that leads to it.
        self.parents           = [-1]                                        # parent index of each node in the RRT tree.
        self.clock_time        = [0.]                                        # clock time at which each node was found.
        self.iteration_count   = [0]                                         # number of iterations after which the node was found.
        

        self.biased_sampler   : BiasedSampler   = None
        self.unbiased_sampler : UnbiasedSampler = UnbiasedSampler(self.map.workspace)
        
        self.spline_connector : SplineConnector = SplineConnector(dim = self.map.workspace.num_dimensions, stl_constraints = self.stl_constraints) # create a spline connector for the STL constraints
        self.max_task_time    : float           = max(stl_constraints, key=lambda x: x.end_time).end_time
        self.kd_tree_past     : KDTree          = KDTree([self.start_node]) 
        self.bias_future_time : bool            = bias_future_time
        
        
        # For a given node : state[self.STATE] is the state and state[self.TIME] is the time
        self.TIME  = self.map.workspace.num_dimensions
        self.STATE = [i for i in range(self.map.workspace.num_dimensions)]
        

        self.new_nodes     : list[np.ndarray]  = []  
        self.sampled_nodes : list[np.ndarray]  = []  
        self.solutions     : list[RRTSolution] = []  

        self.iteration     : int  = 0
        self.verbose       : bool = verbose

        self.rewiring_ratio   : float  = rewiring_ratio
        self.kd_tree_future   : KDTree = KDTree([self.start_node]) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)
        

        # Counter for statistics
        self.successful_rewiring_count  : int = 0
        self.failed_rewiring_count      : int = 0
        self.successful_steering_count  : int = 0
        self.failed_steering_count      : int = 0
        self.collisions_count           : int = 0
        
        if rewiring_radius == -1:
            self.rewiring_radius = 5*self.space_step_size
        else :
            self.rewiring_radius = rewiring_radius
        
        if biasing_ratio < 1:
            biasing_ratio = 1
            print("Biasing ratio must be greater than 1. Setting to 1.")
        
        self.biasing_probability = 1/biasing_ratio

        self.start_clock_time = 0.


    def random_node(self):
        """Generate random node"""
        
        # sample from the bias sampler
        value = np.random.choice([True, False],1, p=[self.biasing_probability, 1-self.biasing_probability])
        if value:
            random_node = self.biased_sampler.get_sample(max_time = self.max_task_time*1.5)
        else:
            random_node = self.unbiased_sampler.get_sample(max_time = self.max_task_time*1.5)
        
        return  random_node
    
    def single_past_nearest(self, node):
        """
        Find the nearest node.
        """
        
        nearest_index : int        = self.kd_tree_past.query(node,k=1)[1]
        nearest_node  : np.ndarray = self.tree[nearest_index]
        return nearest_node, nearest_index
    
    def steer(self, from_node : np.ndarray, to_node : np.ndarray):
        """
        Steer from one node towards another while checking barrier constraints.
        """
        

        from_time  = from_node[self.TIME] 
        from_state = from_node[self.STATE]
        to_time    = to_node[self.TIME]
        to_state   = to_node[self.STATE]

        x_trj,cost = self.spline_connector.connect(x0 = from_state,x1= to_state, t0 = from_time, t1 = to_time)  # connect the two nodes with a spline

        new_node   = to_node
        
        is_in_collision = self.is_trajectory_in_collision(x_trj)
       
        return new_node, x_trj, is_in_collision, cost

    
    def step(self):

        random_node        = self.random_node() # ne sampled node in space-time
        past_tree          = [ node if node[self.TIME] < random_node[self.TIME] else node*1e6 for node in self.tree ]
        self.kd_tree_past  = KDTree(past_tree) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)

        nearest_node,nearest_index  = self.single_past_nearest(random_node)

        # Move towards the random point with limited direction
        nearest_node_state  = nearest_node[self.STATE]
        
        # take a step of maximu  distance from the nearest node towards the random node
        random_node_state   = random_node[self.STATE]
        direction           = random_node_state - nearest_node_state
        direction           = self.space_step_size * direction / np.linalg.norm(direction)
        
        random_node_state   = nearest_node_state + direction
        
        
        random_node         = np.hstack((random_node_state, random_node[self.TIME])) # random node with maximum distance limit
        
        
        self.sampled_nodes.append( random_node)  
        
        try:
            new_node, traj, is_in_collision,cost  = self.steer(nearest_node, random_node )
        except Exception as e:
            raise Exception(f"Error in steering at iteration {self.iteration}, with exception: {e}")

        if not is_in_collision:
            self.tree.append(new_node)
            self.parents.append(nearest_index)
            self.cost.append(self.cost[nearest_index] + cost)
            self.trajectories.append(traj)   
            node_time = time.perf_counter() - self.start_clock_time
            self.clock_time.append(node_time) 
            self.iteration_count.append(self.iteration)
        else :
            self.collisions_count += 1
        
    
    def plan(self):
        """
        Run the RRT algorithm to find a path from start to goal with time constraints and barrier function.
        
        """
        
        self.start_clock_time = time.perf_counter()
        for iteration in tqdm(range(self.max_iter)):
            self.iteration = iteration
            try:
                self.step()
                self.successful_steering_count += 1
            except Exception as e:
                self.failed_steering_count += 1
                if self.verbose:
                    print(f"Error in planning at iteration {iteration}, with exception: {e}")
                continue
            

            if iteration % self.rewiring_ratio  == 0:
                rewiring_candidates = self.get_candidate_rewiring_set()
                for candidate_index in rewiring_candidates:
                    try:
                        self.rewire(candidate_index)
                        self.successful_rewiring_count += 1
                    except Exception as e:
                        self.failed_rewiring_count += 1
                        if self.verbose:
                            print(f"Error in rewiring at iteration {iteration}, with exception: {e}")
                        continue       
        
        self.solutions = self.get_solutions()
        
        print("=============================================")
        print("Solution Summary:")
        print("=============================================")
        for solution in self.solutions:
            print("Cost: %.5f"%solution["cost"])
            print("Clock time: %.2f"%solution["clock_time"])
            print("Number of nodes: %d"%len(solution["path_trj"]))
            print("Number of iterations: %d"%solution["iter"])
            print("----------------------------------------------")
        
        
        
        
        if len(self.solutions) == 0:
            print("No solutions found.")

            stats = {"best_sol_cost"        : None,
                     "best_sol_clock_time"  : None,
                     "first_sol_cost"       : None,
                     "first_sol_clock_time" : None}
            

        else:
            best_solution : RRTSolution = min(self.solutions, key=lambda x: x["cost"])
            print("Best solution*:")
            print("Best solution cost*: %.5f"%best_solution["cost"])
            print("Best solution clock time*: %.2f"%best_solution["clock_time"])
            print("Best solution number of nodes*: %d"%len(best_solution["path_trj"]))
            print("Best solution number of iterations*: %d"%best_solution["iter"])
            print("=============================================")


            stats = {"best_sol_cost"       : best_solution["cost"],
                     "best_sol_clock_time" : best_solution["clock_time"],
                     "first_sol_cost"      : self.solutions[0]["cost"],
                     "first_sol_clock_time" : self.solutions[0]["clock_time"],}
        
        
        return self.solutions, stats
    

    def get_solutions(self):

        terminal_nodes_index_pairs = [(i,node) for i,node in enumerate(self.tree) if node[self.TIME] >= self.max_task_time]
        
        solutions = []
        for index_t, node_t in terminal_nodes_index_pairs:
            if self.cost[index_t] < self.current_best_cost:
                index     = index_t
                path_traj = []
                nodes     = []
                self.current_best_cost = self.cost[index_t]
                while index != -1:
                    path_traj.append(self.trajectories[index])
                    nodes.append(self.tree[index])
                    index  = self.parents[index]

                
                path_traj.reverse()
                nodes.reverse()

                path_solution = RRTSolution()
                path_solution["path_trj"]     = path_traj
                path_solution["cost"]         = self.cost[index_t] 
                path_solution["nodes"]        = nodes
                path_solution["clock_time"]   = self.clock_time[index_t]
                path_solution["iter"]         = self.iteration_count[index_t]

                solutions.append(path_solution)
        return solutions
    
    def get_best_solution(self):
        """
        Get the best solution from the RRT tree.
        """
        if len(self.solutions) == 0:
            print("No solutions found.")
            return None
        else:
            best_solution : RRTSolution = min(self.solutions, key=lambda x: x["cost"])
            return best_solution
        
    
    def is_trajectory_in_collision(self, x_trj : BezierCurve) -> bool:
        for x in x_trj.control_points:
            for obstacle in self.map.obstacles_inflated:
                if x in obstacle:
                    return True
        return False
         
    # def plot_rrt_solution(self, solution_only:bool = False, projection_dim:list[int] = [], ax = None, legend = False):

    #     # Remainder of the class remains the same (plot and animate methods)

        
    #     if len(self.solutions) :
    #         best_solution : RRTSolution = min(self.solutions, key=lambda x: x["cost"])
    #         # best_smoothen : RRTSolution = self.smoothen_solution(best_solution, smoothing_window = 8)
    #         best_smoothen = best_solution
        
    #     else:
    #         print("RRT failed to find a solution. Showing only the achived tree")

        
    #     if len(projection_dim) == 0:
    #         projection_dim = [i for i in range(min(self.system.size_state, 3))] # default projection on the first tree/two dimensions
        

    #     if ax is None:
    #         fig,ax = self.map.draw(projection_dim = projection_dim) # creats already axes of good dimension and throws errors if the number of projected dimensions is wrong
    #     else:
    #         fig = ax.figure

        
    #     C = selection_matrix_from_dims(self.system.size_state, projection_dim)
        
    #     if not solution_only:
    #         # plot the whole tree
    #         for i, node in enumerate(self.tree):
    #             if self.parents[i] != -1:
    #                 trajectory = C@self.trajectories[i]
                    
    #                 if len(projection_dim) == 2:
    #                     x = trajectory[0,:]
    #                     y = trajectory[1,:]
    #                     ax.plot(x, y, "b-o", lw=1)
                    
    #                 elif len(projection_dim) == 3:
    #                     x = trajectory[0,:]
    #                     y = trajectory[1,:]
    #                     z = trajectory[2,:]
    #                     ax.plot(x, y, z, "b-o", lw=1)
        
    #     if not solution_only:
    #         for solution in self.solutions:
    #             for jj,trj in enumerate(solution["path_trj"]):
    #                 trj = C@trj
    #                 if len(projection_dim) == 2:
    #                     x = trj[0,:]
    #                     y = trj[1,:]
    #                 elif len(projection_dim) == 3:
    #                     x = trj[0,:]
    #                     y = trj[1,:]
    #                     z = trj[2,:]
                    
    #                 if jj ==1:
    #                     # plot 
    #                     if len(projection_dim) == 2:
    #                         ax.plot(x, y, lw=4, c = "k", label="Cost: %.5f"%solution["cost"] + "Clock time: %.2f"%solution["clock_time"])
    #                     elif len(projection_dim) == 3:
    #                         ax.plot(x, y, z, lw=4, c = "k", label="Cost: %.5f"%solution["cost"]+ "Clock time: %.2f"%solution["clock_time"])
    #                 else :
    #                     # plot 
    #                     if len(projection_dim) == 2:
    #                         ax.plot(x, y, lw=4, c = "k")
    #                     elif len(projection_dim) == 3:
    #                         ax.plot(x, y, z, lw=4, c = "k")
                        
                    
    #                 # annotate the time at final point 
    #                 # ax.annotate(f"t: {time[-1]:.2f}", (x[-1] +0.3 , y[-1]), textcoords="offset points", xytext=(0,10), ha='center')
    #             # find best trajectory in terms of cost

    #     if len(self.solutions) :
    #         # === Step 1: Concatenate the whole trajectory === #
    #         all_trj = []

    #         for trj in best_smoothen["path_trj"]:
    #             trj = C @ trj
    #             all_trj.append(trj)

    #         # Concatenate along time (axis=1)
    #         full_trj   = np.concatenate(all_trj, axis=1)
    #         total_time = (len(best_smoothen["path_trj"])-1)*self.delta_t

    #         # === Step 2: Extract and plot with global time-based coloring === #
    #         if len(projection_dim) == 2:
    #             x = full_trj[0, :]
    #             y = full_trj[1, :]

    #             points = np.array([x, y]).T.reshape(-1, 1, 2)
    #             segments = np.concatenate([points[:-1], points[1:]], axis=1)

    #             t_values = np.linspace(0, total_time, len(x) - 1)

    #             lc = LineCollection(segments, cmap='cool', array=t_values, linewidth=4)
    #             ax.add_collection(lc)

    #         elif len(projection_dim) == 3:
    #             x = full_trj[0, :]
    #             y = full_trj[1, :]
    #             z = full_trj[2, :]

    #             points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    #             segments = np.concatenate([points[:-1], points[1:]], axis=1)

    #             t_values = np.linspace(0, total_time, len(x) - 1)
                
    #             lc = Line3DCollection(segments, cmap='cool', array=t_values, linewidth=4)
    #             ax.add_collection3d(lc)


    #         plt.colorbar(lc, ax=ax, label='Time progression [s]')

    #     if legend :  
    #         ax.legend()
        
    #     return fig, ax
    
    # def show_statistics(self):
        
    #     success_steer_percentage  = self.successful_steering_count / (self.successful_steering_count + self.failed_steering_count) * 100
    #     collision_percentage      = self.collisions_count /self.max_iter * 100
    #     if self.successful_rewiring_count + self.failed_rewiring_count == 0:
    #         success_rewire_percentage = 100
    #     else:
    #         success_rewire_percentage = self.successful_rewiring_count / (self.successful_rewiring_count + self.failed_rewiring_count) * 100

    #     failed_steer_percentage  = 100 - success_steer_percentage
    #     failed_rewire_percentage = 100 - success_rewire_percentage

    #     category = ('Steer','Rewire')
    #     category_count = {
    #         'Successful': np.array([success_steer_percentage,success_rewire_percentage]),
    #         'Failed': np.array([failed_steer_percentage,failed_rewire_percentage]),
    #     }
    #     width = 0.6  # the width of the bars: can also be len(x) sequence

    #     fig, ax = plt.subplots()
    #     bottom = np.zeros(2)

    #     for outcome,count in category_count.items():
    #         p = ax.bar(category, count, width, label= outcome, bottom=bottom)
    #         bottom += count

    #         ax.bar_label(p, label_type='center')

    #     ax.set_title('RRT statistics')
    #     ax.legend()

    #     category = ('Collision events')
    #     category_count = {
    #         'collision': np.array([collision_percentage]),
    #         'good'     : np.array([100-collision_percentage]),
    #     }
    #     width = 0.6  # the width of the bars: can also be len(x) sequence

    #     fig, ax = plt.subplots()
    #     bottom = np.zeros(2)

    #     for outcome,count in category_count.items():
    #         p = ax.bar(category, count, width, label= outcome, bottom=bottom)
    #         bottom += count

    #         ax.bar_label(p, label_type='center')

    #     ax.set_title('Collision statistics')
    #     ax.legend()
    

    
    # def is_trajectory_in_collision(self, x_trj):
    #     for x in x_trj.T:
    #         for obstacle in self.map.obstacles_inflated:
    #             if x in obstacle:
    #                 return True
    #     return False