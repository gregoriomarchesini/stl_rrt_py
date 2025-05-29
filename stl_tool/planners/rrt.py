
import numpy as np
import matplotlib.pyplot as plt
from   typing          import TypedDict
from   tqdm            import tqdm
from   matplotlib.collections import LineCollection
from   mpl_toolkits.mplot3d.art3d import Line3DCollection
from   scipy.spatial   import KDTree
from   scipy.interpolate import BSpline
import time

from ..openmpc.mpc     import TimedMPC, MPCProblem
from ..openmpc.models  import LinearSystem
from ..openmpc.support import TimedConstraint
from ..openmpc.models  import LinearSystem


from stl_tool.environment.map  import Map
from stl_tool.polyhedron         import Polyhedron,selection_matrix_from_dims
from stl_tool.stl.parameter_optimizer import TimeVaryingConstraint
from stl_tool.stl.linear_system import ContinuousLinearSystem



BIG_NUMBER = 1e10
     
class RRTSolution(TypedDict):
    path_trj     : list[np.ndarray]
    cost         : float
    iter         : int
    nodes        : list[np.ndarray]
    clock_time   : float


class BiasedSampler:
    def __init__(self, list_of_polytopes : list[Polyhedron], list_of_times : list[float]):
        
        
        self.list_of_polytopes :list[Polyhedron] = list_of_polytopes
        self.list_of_times     :list[float]    = list_of_times

        if len(list_of_polytopes) != len(list_of_times):
            raise ValueError("The number of polytopes and times must be the same.")
    
    def get_sample(self, max_time :float) -> np.ndarray :
        
        

        # Choose randomly among them
        random_index = np.random.choice(len(self.list_of_polytopes))
        t_tilde  :float       = self.list_of_times[random_index]
        polytope :Polyhedron    = self.list_of_polytopes[random_index]
        x_tilde  :np.ndarray  = polytope.sample_random()
        
        return np.hstack((x_tilde.flatten(),t_tilde)) # return the sample in the form of (x,y,t)
    
class UnbiasedSampler:
    def __init__(self,workspace : Polyhedron):
        self.workspace = workspace

    def get_sample(self, max_time :float) -> np.ndarray :
        t_tilde  = np.random.uniform(0, max_time)
        x_tilde  :np.ndarray  = self.workspace.sample_random()
        return np.hstack((x_tilde.flatten(),t_tilde)) # return the sample in the form of (x,y,t)
          
class StlRRTStar :
    def __init__(self, start_state      : np.ndarray, 
                       system           : ContinuousLinearSystem,
                       prediction_steps : int,
                       stl_constraints  : list[TimeVaryingConstraint],
                       max_input        : float,
                       map              : Map,
                       max_iter         : int   = 100000, 
                       space_step_size  : float = 0.3, 
                       time_step_size   : float = 0.1,
                       bias_future_time : bool  = True,
                       verbose          : bool  = False ,
                       rewiring_radius  : float = -1,
                       rewiring_ratio   : int   = 2,
                       biasing_ratio    : int   = 2) -> None :
        
        
        start_state            = np.array(start_state).flatten()
        self.system            = LinearSystem.c2d(system.A, system.B, dt = system.dt) # convert to discrete time system of openmpc
        self.start_time        = 0.                                                   # initial time
        self.start_cost        = 0.                                                   # start cost of the initial node
        self.start_node        = np.array([*start_state, self.start_time])            # Start node (node = (state,time))
        self.map               = map                                                  # map containing the obstacles
        self.max_iter          = max_iter                                             # maximum number of iterations
        self.space_step_size   = space_step_size                                      # space time size
        self.time_step_size    = time_step_size                                       # maximum step size of propagation for the MPC
        self.space_time_dist   = np.sqrt(self.space_step_size**2 + self.time_step_size**2) # distance in the nodes domain
        self.prediction_steps  = prediction_steps
        self.max_input_bound   = max_input
        self.stl_constraints   = stl_constraints

        
        # initial checks before moving on
        if len(start_state) != system.size_state:
            raise ValueError(f"Start state dimension {len(start_state)} does not match system state dimension {system.size_state}.")
        
        if not start_state in self.map.workspace:
            raise ValueError(f"Start state {start_state} is not in the workspace.")
        

        # Create RRT list
        self.tree              = [self.start_node]
        self.current_best_cost = BIG_NUMBER
        self.cost              = [0.]
        self.trajectories      = [start_state.flatten()[:,np.newaxis]]
        self.time_trajectories = [self.start_time]
        self.parents           = [-1]
        self.clock_time        = [0.] # clock time at which each node was found
        self.iteration_count   = [0]
        

        self.biased_sampler   : BiasedSampler   = None
        self.unbiased_sampler : UnbiasedSampler = UnbiasedSampler(self.map.workspace)
        
        self.expand_mpc        = self._get_mpc_controller_for_expansion() # MPC controller for the RRT expansion
        self.max_task_time     = max(stl_constraints, key=lambda x: x.end_time).end_time
        self.kd_tree_past      = KDTree([self.start_node]) 
        self.bias_future_time  = bias_future_time
        
        
        # For a given node : state[self.STATE] is the state and state[self.TIME] is the time
        self.TIME  = self.system.size_state
        self.STATE = [i for i in range(self.system.size_state)]
        

        self.new_nodes        :list[np.ndarray]  = []  
        self.sampled_nodes    :list[np.ndarray]  = []  
        self.solutions        :list[RRTSolution] = []  

        self.iteration         = 0
        self.verbose           = verbose

        self.time_sequence           = np.linspace(0., self.max_task_time*1.5,10)
        self.current_time_bias_index = 0
        

        self.rewire_controllers = {i: self._get_mpc_controller_for_rewiring(steps=i) for i in range(1,10)}

        self.delta_t = self.prediction_steps * self.system.dt

        self.rewiring_ratio    = rewiring_ratio
        self.kd_tree_future    = KDTree([self.start_node]) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)
        

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

    def _get_mpc_controller_for_expansion(self) -> TimedMPC:
        """
        Get the MPC controller for the RRT expansion
        """

        print("Initializing MPC controller for tree expansion...")
        # Define MPC parameters
        Q = np.eye(self.system.size_state) * 10  # State penalty matrix
        R = np.eye(self.system.size_input) * 1   # Input penalty matrix

        # Create MPC parameters object
        mpc_params = MPCProblem(system  = self.system, 
                                horizon = self.prediction_steps, 
                                Q       = Q, 
                                R       = R, 
                                solver  = "MOSEK",
                                slack_penalty = "LINEAR")

        # Add input magnitude constraint (elevator angle limited to ±15°)
        mpc_params.add_input_magnitude_constraint(limit = self.max_input_bound, is_hard=True)
        mpc_params.add_general_state_constraints(Hx = self.map.workspace.A, bx = self.map.workspace.b,is_hard=True)

        # convertion into openmpc constraints
        rrt_constraints    :list[TimedConstraint]        = []
        for tvc in self.stl_constraints:
            rrt_constraints.append(TimedConstraint(H     = tvc.H,
                                                   b     = tvc.b, 
                                                   start = tvc.start_time,
                                                   end   = tvc.end_time))
    

        for constraint in rrt_constraints :
            mpc_params.add_general_state_time_constraints(Hx = constraint.H, bx = constraint.b, start_time = constraint.start, end_time = constraint.end, is_hard=True)
        
        mpc_params.reach_refererence_at_steady_state(False) # allows the reference to bereached without being in steady state
        mpc_params.soften_tracking_constraint(penalty_weight = 10.)

        # Create the MPC object
        mpc = TimedMPC(mpc_params)

        # create biased sampler by looking at the polytope defined by each constraint at the final time of satisfaction.
        polytope_times_pairs = []
        for tvc in rrt_constraints:
            # constraint is H[x,t] <= b which at time final_time becomes H[x] <= b - h*final_time 
            H = tvc.H[:,:-1]
            h = tvc.H[:,-1]
            b = tvc.b
            final_time = tvc.end
            polytope = Polyhedron(H, b-h*final_time)
            polytope_times_pairs.append((polytope, final_time))

        self.biased_sampler = BiasedSampler(list_of_polytopes = [polytope for polytope, time in polytope_times_pairs],
                                            list_of_times     = [time for polytope, time in polytope_times_pairs])
        
        return mpc
    
    def _get_mpc_controller_for_rewiring(self,steps = 1) -> TimedMPC:
        """
        Get the MPC controller for the RRT rewiring
        """

        print("Initializing MPC controller for tree rewiring...")
        # Define MPC parameters
        Q = np.eye(self.system.size_state) * 10  # State penalty matrix
        R = np.eye(self.system.size_input) * 1   # Input penalty matrix

        # Create MPC parameters object
        mpc_params = MPCProblem(system  = self.system, 
                                horizon = self.prediction_steps*steps, 
                                Q       = Q, 
                                R       = R, 
                                solver  = "MOSEK",
                                slack_penalty= "LINEAR")

        # Add input magnitude constraint (elevator angle limited to ±15°)
        mpc_params.add_input_magnitude_constraint(limit = self.max_input_bound, is_hard=True)
        mpc_params.add_general_state_constraints(Hx = self.map.workspace.A, bx = self.map.workspace.b,is_hard=True)


        # convertion into openmpc constraints
        rrt_constraints    :list[TimedConstraint]        = []
        for tvc in self.stl_constraints:
            rrt_constraints.append(TimedConstraint(H     = tvc.H,
                                                   b     = tvc.b, 
                                                   start = tvc.start_time,
                                                   end   = tvc.end_time))
            
        for constraint in rrt_constraints :
            mpc_params.add_general_state_time_constraints(Hx = constraint.H, bx = constraint.b, start_time = constraint.start, end_time = constraint.end, is_hard=True)
        
        
        # terminal state constraint 
        mpc_params.reach_refererence_at_steady_state(False) # allows the reference to bereached without being in steady state

        # Create the MPC object
        mpc = TimedMPC(mpc_params)

        return mpc

    
    def get_candidate_rewiring_set(self) -> list[int]:
        # Find the neighbors of the new node within a certain radius
        last_node_index = len(self.tree) - 1
        last_node       =  self.tree[last_node_index] # take out the last node

        future_nodes        = [node  if node[self.TIME] > last_node[self.TIME]  else node*1E8  for node in self.tree]
        self.kd_tree_future = KDTree(future_nodes) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)
        nearest_indices     = self.kd_tree_future.query_ball_point(last_node,r= self.rewiring_radius * (np.log(len(self.tree))/len(self.tree))**(1/2)) # nearest neighbour both in time and space
        return nearest_indices



    def random_node(self):
        """ Generate random node"""
        
        # sample from the bias sampler
        value = np.random.choice([True, False],1, p=[self.biasing_probability, 1-self.biasing_probability])
        if value:
            random_node = self.biased_sampler.get_sample(max_time = self.max_task_time*1.5)
        else:
            random_node = self.unbiased_sampler.get_sample(max_time = self.max_task_time*1.5)
        
        return  random_node
    
    def single_past_nearest(self, node):
        """Find the nearest node."""
        
        nearest_index = self.kd_tree_past.query(node,k=1)[1]
        nearest_node  = self.tree[nearest_index]
        return nearest_node, nearest_index
    
    def steer(self, from_node : np.ndarray, to_node : np.ndarray):
        """Steer from one node towards another while checking barrier constraints."""
        

        from_time  = from_node[self.TIME] 
        from_state = from_node[self.STATE]
        to_state   = to_node[self.STATE]

        #  Get trajectory between `from_node` and `to_node` in time `t_max`
        x_trj, u_trj,to_time = self.expand_mpc.get_state_and_control_trajectory(x0 = from_state ,t0 = from_time, reference = to_state)
        
        new_node        = np.hstack((x_trj[:,-1], to_time))
        
        is_in_collision = self.is_trajectory_in_collision(x_trj)
        cost = np.sum(np.linalg.norm(np.diff(x_trj,axis=1),axis=0))   
        
        return new_node, x_trj, is_in_collision, cost

    
    
    def step(self):

        random_node         = self.random_node()
        past_tree          = [ node if node[self.TIME] < random_node[self.TIME] else node*1e6 for node in self.tree ]
        self.kd_tree_past  = KDTree(past_tree) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)

        nearest_node,nearest_index  = self.single_past_nearest(random_node)

        # Move towards the random point with limited direction
        nearest_node_state  = nearest_node[self.STATE]
        random_node_state   = random_node[self.STATE]
        direction           = random_node_state - nearest_node_state
        direction           = self.space_step_size * direction / np.linalg.norm(direction)
        random_node_state   = nearest_node_state + direction
        random_node         = np.hstack((random_node_state, random_node[self.TIME])) 
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
        
    
    def rewire(self, candidate_index : int) :

        """Rewire the tree with the new node."""
        last_node_index = len(self.tree) - 1
        last_node       =  self.tree[last_node_index] # take out the last node
        
    
    
        # do not recheck your own parent
        if candidate_index == self.parents[last_node_index] :
            return 

        neighbour        = self.tree[candidate_index]

        last_node_state  = last_node[self.STATE]
        last_node_time   = last_node[self.TIME]
          
        neighbour_state  = neighbour[self.STATE]
        neighbour_time   = neighbour[self.TIME]

        
        # this is the integer defining the difference in number of nodes to reach last node and to reach the neighbour. 
        # This also defines the difference in time between the two since the MPC is runned at a fixed time steo
        number_of_nodes_difference = int((neighbour_time-last_node_time)/self.delta_t) # also the number of nodes difference is always going to be greater than 1 since we only consider neighbours in the future.

        if not number_of_nodes_difference in self.rewire_controllers.keys() :
            return

        # Check if the new node is a better parent than the current parent of the neighbour (heuristic)
        expected_cost  = np.linalg.norm(last_node_state - neighbour_state) + self.cost[last_node_index]
        
        # attempt rewiring
        if expected_cost < self.cost[candidate_index]:
            # Rewire the tree
            try :
                #  Get trajectory between `from_node` and `to_node` in time `t_max`
                x_trj, u_trj, new_final_time = self.rewire_controllers[number_of_nodes_difference].get_state_and_control_trajectory(x0 = last_node_state ,t0 = last_node_time, reference = neighbour_state)
            except Exception as e:
                raise Exception(f"Error in rewiring at iteration {self.iteration}, with exception: {e}")
                
            actual_new_cost  = self.cost[last_node_index]
            neighbour_time   = new_final_time
            is_in_collision  = self.is_trajectory_in_collision(x_trj)
            actual_new_cost += np.sum(np.linalg.norm(np.diff(x_trj,axis=1),axis=0))   


            if (not is_in_collision) and (actual_new_cost < self.cost[candidate_index]):
                
                self.tree[candidate_index]         = np.hstack((neighbour_state, new_final_time)) # update the node
                self.parents[candidate_index]      = last_node_index  # the new neighbour is the one that was just added
                self.trajectories[candidate_index] = x_trj
                self.cost[candidate_index]         = actual_new_cost
    

    
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
         
    def plot_rrt_solution(self, solution_only:bool = False, projection_dim:list[int] = [], ax = None, legend = False):

        # Remainder of the class remains the same (plot and animate methods)

        
        if len(self.solutions) :
            best_solution : RRTSolution = min(self.solutions, key=lambda x: x["cost"])
            # best_smoothen : RRTSolution = self.smoothen_solution(best_solution, smoothing_window = 8)
            best_smoothen = best_solution
        
        else:
            print("RRT failed to find a solution. Showing only the achived tree")

        
        if len(projection_dim) == 0:
            projection_dim = [i for i in range(min(self.system.size_state, 3))] # default projection on the first tree/two dimensions
        

        if ax is None:
            fig,ax = self.map.draw(projection_dim = projection_dim) # creats already axes of good dimension and throws errors if the number of projected dimensions is wrong
        else:
            fig = ax.figure

        
        C = selection_matrix_from_dims(self.system.size_state, projection_dim)
        
        if not solution_only:
            # plot the whole tree
            for i, node in enumerate(self.tree):
                if self.parents[i] != -1:
                    trajectory = C@self.trajectories[i]
                    
                    if len(projection_dim) == 2:
                        x = trajectory[0,:]
                        y = trajectory[1,:]
                        ax.plot(x, y, "b-o", lw=1)
                    
                    elif len(projection_dim) == 3:
                        x = trajectory[0,:]
                        y = trajectory[1,:]
                        z = trajectory[2,:]
                        ax.plot(x, y, z, "b-o", lw=1)
        
        if not solution_only:
            for solution in self.solutions:
                for jj,trj in enumerate(solution["path_trj"]):
                    trj = C@trj
                    if len(projection_dim) == 2:
                        x = trj[0,:]
                        y = trj[1,:]
                    elif len(projection_dim) == 3:
                        x = trj[0,:]
                        y = trj[1,:]
                        z = trj[2,:]
                    
                    if jj ==1:
                        # plot 
                        if len(projection_dim) == 2:
                            ax.plot(x, y, lw=4, c = "k", label="Cost: %.5f"%solution["cost"] + "Clock time: %.2f"%solution["clock_time"])
                        elif len(projection_dim) == 3:
                            ax.plot(x, y, z, lw=4, c = "k", label="Cost: %.5f"%solution["cost"]+ "Clock time: %.2f"%solution["clock_time"])
                    else :
                        # plot 
                        if len(projection_dim) == 2:
                            ax.plot(x, y, lw=4, c = "k")
                        elif len(projection_dim) == 3:
                            ax.plot(x, y, z, lw=4, c = "k")
                        
                    
                    # annotate the time at final point 
                    # ax.annotate(f"t: {time[-1]:.2f}", (x[-1] +0.3 , y[-1]), textcoords="offset points", xytext=(0,10), ha='center')
                # find best trajectory in terms of cost

        if len(self.solutions) :
            # === Step 1: Concatenate the whole trajectory === #
            all_trj = []

            for trj in best_smoothen["path_trj"]:
                trj = C @ trj
                all_trj.append(trj)

            # Concatenate along time (axis=1)
            full_trj   = np.concatenate(all_trj, axis=1)
            total_time = (len(best_smoothen["path_trj"])-1)*self.delta_t

            # === Step 2: Extract and plot with global time-based coloring === #
            if len(projection_dim) == 2:
                x = full_trj[0, :]
                y = full_trj[1, :]

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                t_values = np.linspace(0, total_time, len(x) - 1)

                lc = LineCollection(segments, cmap='cool', array=t_values, linewidth=4)
                ax.add_collection(lc)

            elif len(projection_dim) == 3:
                x = full_trj[0, :]
                y = full_trj[1, :]
                z = full_trj[2, :]

                points = np.array([x, y, z]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                t_values = np.linspace(0, total_time, len(x) - 1)
                
                lc = Line3DCollection(segments, cmap='cool', array=t_values, linewidth=4)
                ax.add_collection3d(lc)


            plt.colorbar(lc, ax=ax, label='Time progression [s]')

        if legend :  
            ax.legend()
        
        return fig, ax
    
    def show_statistics(self):
        
        success_steer_percentage  = self.successful_steering_count / (self.successful_steering_count + self.failed_steering_count) * 100
        collision_percentage      = self.collisions_count /self.max_iter * 100
        if self.successful_rewiring_count + self.failed_rewiring_count == 0:
            success_rewire_percentage = 100
        else:
            success_rewire_percentage = self.successful_rewiring_count / (self.successful_rewiring_count + self.failed_rewiring_count) * 100

        failed_steer_percentage  = 100 - success_steer_percentage
        failed_rewire_percentage = 100 - success_rewire_percentage

        category = ('Steer','Rewire')
        category_count = {
            'Successful': np.array([success_steer_percentage,success_rewire_percentage]),
            'Failed': np.array([failed_steer_percentage,failed_rewire_percentage]),
        }
        width = 0.6  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()
        bottom = np.zeros(2)

        for outcome,count in category_count.items():
            p = ax.bar(category, count, width, label= outcome, bottom=bottom)
            bottom += count

            ax.bar_label(p, label_type='center')

        ax.set_title('RRT statistics')
        ax.legend()

        category = ('Collision events')
        category_count = {
            'collision': np.array([collision_percentage]),
            'good'     : np.array([100-collision_percentage]),
        }
        width = 0.6  # the width of the bars: can also be len(x) sequence

        fig, ax = plt.subplots()
        bottom = np.zeros(2)

        for outcome,count in category_count.items():
            p = ax.bar(category, count, width, label= outcome, bottom=bottom)
            bottom += count

            ax.bar_label(p, label_type='center')

        ax.set_title('Collision statistics')
        ax.legend()
    

    def smoothen_solution(self,solution : RRTSolution, smoothing_window : int = 4):

        smoothing_window = int(smoothing_window)
        N                = self.prediction_steps
        if smoothing_window not in self.rewire_controllers:
            raise ValueError(
                f"Smoothing window {smoothing_window} is not in range. "
                f"Available windows: {list(self.rewire_controllers.keys())}"
            )

        nodes = solution["nodes"]
        trjs = solution["path_trj"]

        new_nodes   = nodes.copy()
        new_trjs    = trjs.copy()  # Initialize empty trajectory list
        new_trjs[0] = trjs[0]  # Keep first trajectory as is

        i = 0
        while i < len(nodes) - 1:
            max_window = min(smoothing_window, len(nodes) -1 - i)

            from_node  = new_nodes[i]
            to_node    = new_nodes[i + max_window]
            controller = self.rewire_controllers[max_window]
                
            try:
                # Get trajectory that exactly connects these nodes
                x_trj, u_trj, _ = controller.get_state_and_control_trajectory(
                    x0=from_node[self.STATE],
                    t0=from_node[self.TIME],
                    reference=to_node[self.STATE]
                )
            except Exception as e:
                print(f"Smoothing failed between {i} and {i+max_window}: {e}")
                i +=1 
                continue

            new_cost = np.sum(np.linalg.norm(np.diff(x_trj, axis=1), axis=0))
            # old cost
            old_cost = sum(np.sum(np.linalg.norm(np.diff(trj, axis=1), axis=0)) for trj in trjs[i:i + max_window + 1])


            # Check collision for the entire trajectory
            if not self.is_trajectory_in_collision(x_trj) and (new_cost <= old_cost):
                # Update intermediate nodes
                for k in range(1, max_window):
                    t_k      = from_node[self.TIME] + k * self.delta_t
                    state_k  = x_trj[:,k*N]
                    mid_trj  = x_trj[:,(k-1)*N : k*N+1]
                    
                    
                    # Find the exact point in the trajectory at time t_k
                    new_nodes[i + k] = np.hstack((state_k, t_k))
                    new_trjs[i + k]  = mid_trj
                
            else :
                # If collision, keep the original trajectory
                new_trjs[i + 1:i + max_window] = trjs[i + 1:i + max_window]
                new_nodes[i + 1:i + max_window] = nodes[i + 1:i + max_window]
            
            i +=1 

            
        # Recompute cost
        cost = sum(
            np.sum(np.linalg.norm(np.diff(trj, axis=1), axis=0)) for trj in new_trjs[1:]  ) # xplore first trajectory that is juts a point

        new_sol = RRTSolution()
        new_sol["cost"]     = cost
        new_sol["iter"]     = self.iteration
        new_sol["path_trj"] = new_trjs
        new_sol["nodes"]    = new_nodes

        return new_sol

    def is_trajectory_in_collision(self, x_trj):
        for x in x_trj.T:
            for obstacle in self.map.obstacles_inflated:
                if x in obstacle:
                    return True
        return False