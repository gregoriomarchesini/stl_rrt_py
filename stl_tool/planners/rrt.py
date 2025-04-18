
import random
import numpy as np
import matplotlib.pyplot as plt
from   typing          import TypedDict
from   tqdm            import tqdm
from   scipy.spatial   import KDTree
from scipy.interpolate import BSpline

from openmpc.mpc     import TimedMPC, MPCProblem
from openmpc.models  import LinearSystem
from openmpc.support import TimedConstraint
from openmpc.models  import LinearSystem


from stl_tool.environment.map  import Map
from stl_tool.polytope         import Polytope,selection_matrix_from_dims
from stl_tool.stl.parameter_optimizer import TimeVaryingConstraint
from stl_tool.stl.linear_system import ContinuousLinearSystem



BIG_NUMBER = 1e10
     
class RRTSolution(TypedDict):
    path_trj     : list
    cost         : float
    iter         : int




class BiasedSampler:
    def __init__(self, list_of_polytopes : list[Polytope], list_of_times : list[float]):
        
        
        self.list_of_polytopes :list[Polytope] = list_of_polytopes
        self.list_of_times     :list[float]    = list_of_times

        if len(list_of_polytopes) != len(list_of_times):
            raise ValueError("The number of polytopes and times must be the same.")
    
    def get_sample(self, max_time :float) -> np.ndarray :
        
        t_tilde  = np.random.uniform(0, max_time) 
        random_index = np.argmin(np.abs(np.array(self.list_of_times) - t_tilde)) # find the index of the closest time
        polytope :Polytope = self.list_of_polytopes[random_index]
        x_tilde  :np.ndarray  = polytope.sample_random()

        return np.hstack((x_tilde.flatten(),t_tilde)) # return the sample in the form of (x,y,t)
    
class UnbiasedSampler:
    def __init__(self,workspace : Polytope):
        self.workspace = workspace

    def get_sample(self, max_time :float) -> np.ndarray :
        t_tilde  = np.random.uniform(0, max_time)
        x_tilde  :np.ndarray  = self.workspace.sample_random()
        return np.hstack((x_tilde.flatten(),t_tilde)) # return the sample in the form of (x,y,t)
          
    
class RRT:
    def __init__(self, start_state      :np.ndarray, 
                       system           :ContinuousLinearSystem,
                       prediction_steps :int,
                       stl_constraints  :list[TimeVaryingConstraint],
                       max_input        :float,
                       map              :Map,
                       max_task_time    :float,
                       max_iter         :int   = 100000, 
                       space_step_size  :float = 0.3, 
                       time_step_size   :float = 0.1,
                       bias_future_time :bool = True,
                       verbose : bool = False ,
                       sampler : BiasedSampler = None) -> None :
        
        
        start_state            = np.array(start_state).flatten()
        self.system            = LinearSystem.c2d(system.A, system.B, dt = system.dt) # convert to discrete time system of openmpc
        self.start_time        = 0.                                                   # initial time
        self.start_cost        = 0.                                                   # start cost of the initial node

        if len(start_state) != system.size_state:
            raise ValueError(f"Start state dimension {len(start_state)} does not match system state dimension {system.size_state}.")

        self.start_node        = np.array([*start_state, self.start_time])            # Start node (node = (state,time))
        self.map               = map                                                  # map containing the obstacles
        self.max_iter          = max_iter                                             # maximum number of iterations
        self.space_step_size   = space_step_size                                      # space time size
        self.time_step_size    = time_step_size                                       # maximum step size of propagation for the MPC
        self.space_time_dist   = np.sqrt(self.space_step_size**2 + self.time_step_size**2) # distance in the nodes domain
        self.prediction_steps  = prediction_steps
        self.max_input_bound   = max_input
        self.stl_constraints   = stl_constraints

        self.tree              = [self.start_node]
        self.current_best_cost = BIG_NUMBER
        self.cost              = [0.]
        self.trajectories      = [start_state.flatten()[:,np.newaxis]]
        self.time_trajectories = [self.start_time]
        self.parents           = [-1]
        
        
        self.expand_mpc        = self._get_mpc_controller_for_expansion() # MPC controller for the RRT expansion
        self.max_task_time     = max_task_time
        self.kd_tree_past      = KDTree([self.start_node]) 
        self.bias_future_time  = bias_future_time
        
        
        # For a given node : state[self.STATE] is the state and state[self.TIME] is the time
        self.TIME  = self.system.size_state
        self.STATE = [i for i in range(self.system.size_state)]


        self.successful_steering_count = 0
        self.failed_steering_count     = 0
        

        self.obtained_paths    = []  
        self.new_nodes         = []  
        self.sampled_nodes     = []  
        self.solutions         = []  

        self.iteration         = 0
        self.verbose           = verbose

        self.time_sequence           = np.linspace(0.,max_task_time*1.5,10)
        self.current_time_bias_index = 0

        self.biased_sampler   = sampler
        self.unbiased_sampler = UnbiasedSampler(self.map.workspace)

        # if start if not in the workspace then throw an error
        if not start_state in self.map.workspace:
            raise ValueError(f"Start state {start_state} is not in the workspace.")



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

        return mpc



    def random_node(self):
        """ Generate random node"""
        
        if self.biased_sampler is None:
            random_node = self.unbiased_sampler.get_sample(max_time = self.max_task_time*1.5)
            return random_node
        else:
            # sample from the bias sampler
            value = random.choice([True, False])
            if value:
                random_node = self.biased_sampler.get_sample(max_time = self.max_task_time*1.5)
            else:
                random_node = self.unbiased_sampler.get_sample(max_time = self.max_task_time*1.5)
            
            self.sampled_nodes.append( random_node)  
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
        is_in_collision = False
        cost            = 0.
        
        obstacles : list[Polytope] = self.map.obstacles 
        for ii in range(x_trj.shape[1]): 
            for obstacle in obstacles:
                if x_trj[:,ii] in obstacle:
                    is_in_collision = True
                    break
            
            if ii >=1:
                cost += np.linalg.norm(x_trj[:,ii] - x_trj[:,ii-1])   
        
            
        return new_node, x_trj, is_in_collision, cost

    
    
    def step(self):

        rand_point         = self.random_node()
        past_tree          = [ node if node[self.TIME] < rand_point[self.TIME] else node*1e6 for node in self.tree ]
        self.kd_tree_past  = KDTree(past_tree) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)

        nearest_node,nearest_index  = self.single_past_nearest(rand_point)

        # Move towards the random point with limited direction
        nearest_node_state = nearest_node[self.STATE]
        rand_point_state   = rand_point[self.STATE]
        direction          = rand_point_state - nearest_node_state
        direction          = self.space_step_size * direction / np.linalg.norm(direction)
        rand_point_state   = nearest_node_state + direction
        rand_point         = np.hstack((rand_point_state, rand_point[self.TIME])) 
        
        
        try:
            new_node, traj, is_in_collision,cost  = self.steer(nearest_node, rand_point)
        except Exception as e:
            raise Exception(f"Error in steering at iteration {self.iteration}, with exception: {e}")

        if not is_in_collision:
            self.tree.append(new_node)
            self.parents.append(nearest_index)
            self.cost.append(self.cost[nearest_index] + cost)
            self.trajectories.append(traj)   


    

    
    def plan(self):
        """Run the RRT algorithm to find a path from start to goal with time constraints and barrier function."""
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

        return self.solutions
    

    def get_solution(self):

        terminal_nodes_index_pairs = [(i,node) for i,node in enumerate(self.tree) if node[self.TIME] >= self.max_task_time]

        for index_t, node_t in terminal_nodes_index_pairs:
            if self.cost[index_t] < self.current_best_cost:
                    index     = index_t
                    path_traj = []
                    self.current_best_cost = self.cost[index_t]
                    
                    while index != -1:
                        path_traj.append(self.trajectories[index])
                        index  = self.parents[index]

                    path_solution = RRTSolution()
                    path_solution["path_trj"]     = path_traj
                    path_solution["cost"]         = self.cost[index_t]
                    path_solution["iter"]         = self.iteration 

                    self.solutions.append(path_solution)

         
    def plot_rrt_solution(self, solution_only:bool = False, projection_dim:list[int] = [], ax = None):

        # Remainder of the class remains the same (plot and animate methods)

        
        self.get_solution()

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
                        ax.plot(x, y, lw=4, c = "k", label="Cost: %.5f"%solution["cost"])
                    elif len(projection_dim) == 3:
                        ax.plot(x, y, z, lw=4, c = "k", label="Cost: %.5f"%solution["cost"])
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

            best = min(self.solutions, key=lambda x: x["cost"])

            for jj,trj in enumerate(best["path_trj"]) :
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
                        ax.plot(x, y, lw=4, c = "r", label="Cost: %.5f"%solution["cost"])
                    elif len(projection_dim) == 3:
                        ax.plot(x, y, z, lw=4, c = "r", label="Cost: %.5f"%solution["cost"])
                else :
                    # plot 
                    if len(projection_dim) == 2:
                        ax.plot(x, y, lw=4, c = "r")
                    elif len(projection_dim) == 3:
                        ax.plot(x, y, z, lw=4, c = "r")

        ax.legend()
        
        return fig, ax
    
    def show_statistics(self):

        success_steer_percentage = self.successful_steering_count / (self.successful_steering_count + self.failed_steering_count) * 100
        failed_steer_percentage  = 100 - success_steer_percentage

        category = ('Steer')
        category_count = {
            'Successful': np.array([success_steer_percentage]),
            'Failed': np.array([failed_steer_percentage]),
        }
        width = 0.6  # the width of the bars: can also be len(x) sequence


        fig, ax = plt.subplots()
        bottom = np.zeros(1)

        for outcome,count in category_count.items():
            p = ax.bar(category, count, width, label= outcome, bottom=bottom)
            bottom += count

            ax.bar_label(p, label_type='center')

        ax.set_title('Number of penguins by sex')
        ax.legend()
    

#########################################################################################################################
#########################################################################################################################
class RRTStar(RRT):
    def __init__(self, start_state      :np.ndarray, 
                       system           :LinearSystem,
                       prediction_steps :int,
                       stl_constraints  :list[TimedConstraint],
                       max_input        :float,
                       map              :Map,
                       max_task_time    :float,
                       max_iter         :int   = 100000, 
                       space_step_size  :float = 0.3, 
                       time_step_size   :float = 0.1,
                       bias_future_time :bool  = True,
                       rewiring_radius  :float = -1,
                       rewiring_ratio   :int   = 2,
                       verbose : bool = False) -> None :
        
        super().__init__(start_state, 
                            system ,
                            prediction_steps,
                            stl_constraints,
                            max_input ,
                            map,
                            max_task_time,
                            max_iter, 
                            space_step_size,
                            bias_future_time,
                            verbose = verbose)
        


        self.rewire_controllers = {i: self._get_mpc_controller_for_rewiring(steps=i) for i in range(1,10)}

        self.delta_t = self.prediction_steps * self.system.dt

        self.rewiring_ratio    = rewiring_ratio
        self.kd_tree_future    = KDTree([self.start_node]) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)

        self.successful_rewiring_count  = 0
        self.failed_rewiring_count      = 0
        
        if rewiring_radius == -1:
            self.rewiring_radius = 5*self.space_step_size
        else :
            self.rewiring_radius = rewiring_radius


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

        
        # this is the iteger defining the difference in number of nodes to reach last node and to reach the neighbour. 
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
                

            is_in_collision  = False
            actual_new_cost  = self.cost[last_node_index]
            neighbour_time = new_final_time


            for ii,x in enumerate(x_trj.T): 
                for obstacle in self.map.obstacles:
                    if x in obstacle:
                        is_in_collision = True
                        break
                
                if ii >=1:
                    actual_new_cost += np.linalg.norm(x_trj[:,ii] - x_trj[:,ii-1])   


            if (not is_in_collision) and (actual_new_cost < self.cost[candidate_index]):
                
                self.tree[candidate_index]         = np.hstack((neighbour_state, new_final_time)) # update the node
                self.parents[candidate_index]      = last_node_index  # the new neighbour is the one that was just added
                self.trajectories[candidate_index] = x_trj
                self.cost[candidate_index]         = actual_new_cost

    

    
    def plan(self):
        """Run the RRT algorithm to find a path from start to goal with time constraints and barrier function."""
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

        return self.solutions
    
    def show_statistics(self):
        

        success_steer_percentage  = self.successful_steering_count / (self.successful_steering_count + self.failed_steering_count) * 100
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

        ax.set_title('Number of penguins by sex')
        ax.legend()
    
    



class BisplineRRT(RRT):
    def __init__(self, start_state      :np.ndarray, 
                       system           :ContinuousLinearSystem,
                       prediction_steps :int,
                       stl_constraints  :list[TimeVaryingConstraint],
                       max_input        :float,
                       map              :Map,
                       max_task_time    :float,
                       max_iter         :int   = 100000, 
                       space_step_size  :float = 0.3, 
                       time_step_size   :float = 0.1,
                       bias_future_time :bool = True) -> None :
        
        
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

        self.tree              = [self.start_node]
        self.current_best_cost = BIG_NUMBER
        self.cost              = [0.]
        self.trajectories      = [self.start_node[:,np.newaxis]]
        self.time_trajectories = [self.start_time]
        self.parents           = [-1]
        
        
        self.expand_mpc        = self._get_spline_interplator_for_expansion() # MPC controller for the RRT expansion
        self.max_task_time     = max_task_time
        self.kd_tree_past      = KDTree([self.start_node]) 
        self.bias_future_time  = bias_future_time
        
        
        self.TIME = 2
        self.STATE = [0,1]

        self.obtained_paths    = []  
        self.new_nodes         = []  
        self.sampled_nodes     = []  
        self.solutions         = []  

        self.iteration         = 0



    def _get_spline_interplator_for_expansion(self):
        """
        Get the MPC controller for the RRT expansion
        """

        def generate_bspline(p_start, p_end, t_start, t_end, num_points=20):
            """
            Generate a B-spline with 4 control points between start and end points in space-time
            p_start: starting point in space (x,y,z)
            p_end: ending point in space (x,y,z)
            t_start: starting time
            t_end: ending time
            """
            # Combine space and time dimensions
            start = np.append(p_start, t_start)
            end = np.append(p_end, t_end)
            
            # Create 4 control points (including start and end)
            # You might want to add some intermediate points for smoother curves
            ctrl_pts = np.array([
                start,
                start + 0.33*(end - start),
                start + 0.67*(end - start),
                end
            ])
            
            # Create knot vector for cubic B-spline (degree=3)
            knots = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            
            # Create B-spline object
            degree = 3
            spline = BSpline(knots, ctrl_pts, degree)
            
            # Evaluate at parameter values
            t = np.linspace(0, 1, num_points)
            trajectory = spline(t)
            
            return trajectory[:, :3], trajectory[:, 3]  # space coords, time coords

        return generate_bspline

    
    def steer(self, from_node : np.ndarray, to_node : np.ndarray):
        """Steer from one node towards another while checking barrier constraints."""
        

        from_time  = from_node[self.TIME] 
        from_state = from_node[self.STATE]
        to_state   = to_node[self.STATE]

        #  Get trajectory between `from_node` and `to_node` in time `t_max`
        x_trj, u_trj,to_time = self.expand_mpc.get_state_and_control_trajectory(x0 = from_state ,t0 = from_time, reference = to_state)
        
        new_node        = np.hstack((x_trj[:,-1], to_time))
        is_in_collision = False
        cost            = 0.
        
        obstacles : list[Polytope] = self.map.obstacles 
        for ii in range(x_trj.shape[1]): 
            for obstacle in obstacles:
                if x_trj[:,ii] in obstacle:
                    is_in_collision = True
                    break
            
            if ii >=1:
                cost += np.linalg.norm(x_trj[:,ii] - x_trj[:,ii-1])   
            
        return new_node, x_trj, is_in_collision, cost
    
