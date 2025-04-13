
import random
import numpy as np
import matplotlib.pyplot as plt
from   typing          import TypedDict
from   tqdm            import tqdm
from   scipy.spatial   import KDTree

from openmpc.mpc     import TimedMPC, MPCProblem
from openmpc.models  import LinearSystem
from openmpc.support import TimedConstraint


from stl_tool.environment.map  import Map
from stl_tool.polytope import Polytope


# from src.linear.src.systems import MPCcbfControllerFixedEnd, MPCcbfControllerFreeEnd, LQRcbfControllerLogExp, LQRController
# from src.linear.src.stl     import BarrierFunction,BIG_NUMBER
# from src.linear.src.env     import Map, Obstacle


BIG_NUMBER = 1e10
     
class RRTSolution(TypedDict):
    path_trj     : list
    cost         : float
    iter         : int

    
class RRT:
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
                       bias_future_time :bool = True) -> None :
        
        self.system            = system
        self.start_time        = 0.                                         # initial time
        self.start_cost        = 0.                                         # start cost of the initial node
        self.start_node        = np.array([*start_state, self.start_time])  # Start node (node = (state,time))
        self.map               = map                                        # map containing the obstacles
        self.max_iter          = max_iter                                   # maximum number of iterations
        self.space_step_size   = space_step_size                            # space time size
        self.time_step_size    = time_step_size                             # maximum step size of propagation for the MPC
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
        
        
        self.expand_mpc        = self._get_mpc_controller_for_expansion() # MPC controller for the RRT expansion
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
                                solver  = "MOSEK")

        # Add input magnitude constraint (elevator angle limited to ±15°)
        mpc_params.add_input_magnitude_constraint(limit = self.max_input_bound, is_hard=True)
        mpc_params.add_general_state_constraints(Hx = self.map.workspace.A, bx = self.map.workspace.b,is_hard=True)
        for constraint in self.stl_constraints :
            mpc_params.add_general_state_time_constraints(Hx = constraint.H, bx = constraint.b, start_time = constraint.start, end_time = constraint.end, is_hard=True)
        
        mpc_params.reach_refererence_at_steady_state(False) # allows the reference to bereached without being in steady state
        mpc_params.soften_tracking_constraint(penalty_weight = 10.)

        # Create the MPC object
        mpc = TimedMPC(mpc_params)

        return mpc



    def random_node(self):
        """ Generate random node"""
        if self.bias_future_time:
            x_tilde     = self.map.workspace.sample_random().flatten()
            time_tilde  = random.uniform(self.max_task_time, self.max_task_time*1.5) # sample mostly in the future for fast expansion of the tree
            random_node = np.array([ *x_tilde, time_tilde])
        else :
            x_tilde     = self.map.workspace.sample_random().flatten()
            time_tilde  = random.uniform(self.start_time, self.max_task_time*1.5) # sample mostly in the future for fast expansion of the tree
            random_node = np.array([ *x_tilde, time_tilde])
        
        self.sampled_nodes.append(random_node)  
        return random_node
    
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
        past_tree          = [ node if node[2] < rand_point[2] else node*1e6 for node in self.tree ]
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
            print(f"Error in steering at iteration {self.iteration}, with exception: {e}")

        if not is_in_collision:
            self.tree.append(new_node)
            self.parents.append(nearest_index)
            self.cost.append(self.cost[nearest_index] + cost)
            self.trajectories.append(traj)   


            if new_node[self.TIME] >= self.max_task_time and self.cost[-1] < self.current_best_cost:
                index = len(self.tree) - 1
                path_traj = []
                self.current_best_cost = self.cost[-1]
                
                while index != -1:
                    path_traj.append(self.trajectories[index])
                    index  = self.parents[index]

                path_solution = RRTSolution()
                path_solution["path_trj"]     = path_traj
                path_solution["cost"]         = cost
                path_solution["iter"]         = self.iteration 

                self.solutions.append( path_solution)

    
    def plan(self):
        """Run the RRT algorithm to find a path from start to goal with time constraints and barrier function."""
        for iteration in tqdm(range(self.max_iter)):
            self.iteration = iteration
            try:
                self.step()
            except Exception as e:
                print(f"Error in planning at iteration {iteration}, with exception: {e}")

        return self.solutions


         
    def plot_rrt_solution(self, solution_only:bool = False):

        # Remainder of the class remains the same (plot and animate methods)

        fig,ax = self.map.draw()
        
        if not solution_only:
            # plot the whole tree
            for i, node in enumerate(self.tree):
                if self.parents[i] != -1:
                    trajectory = self.trajectories[i]
                    x = trajectory[0,:]
                    y = trajectory[1,:]

                    ax.plot(x, y, "g-o", lw=1)

        for solution in self.solutions:
            random_color = np.random.rand(3,)
            for j,trj in enumerate(solution["path_trj"]):
                x    = trj[0,:]
                y    = trj[1,:]
                # time = trj[2,:]
                
                if j == len(solution["path_trj"])-1:
                    ax.plot(x, y, lw=4, c = random_color, label="Cost: %.2f"%solution["cost"])
                else :
                    ax.plot(x, y, lw=4, c = random_color)
                # annotate the time at final point 
                # ax.annotate(f"t: {time[-1]:.2f}", (x[-1] +0.3 , y[-1]), textcoords="offset points", xytext=(0,10), ha='center')
           
            # find best trajectory in terms of cost
        if len(self.solutions) :
            best = min(self.solutions, key=lambda x: x["cost"])
            for jj,trj in enumerate(best["path_trj"]):
                x    = trj[0,:]
                y    = trj[1,:]
                # time = trj[2,:]
                if jj == len(best["path_trj"])-1:
                    ax.plot(x, y, lw=4, c = "red", label="Best Cost: %.2f"%best["cost"])
                else :
                    ax.plot(x, y, lw=4, c = "red")
                # annotate the time at final point 
                # ax.annotate(f"t: {time[-1]:.2f}", (x[-1] +0.3 , y[-1]), textcoords="offset points", xytext=(0,10), ha='center', color="red")

            ax.plot(self.start_node[0], self.start_node[1], "go", markersize=10, label="Start")
            ax.legend()
        
        return fig, ax
    



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
                       rewiring_ratio   :int   = 2) -> None :
        
        super().__init__(start_state, 
                            system ,
                            prediction_steps,
                            stl_constraints,
                            max_input ,
                            map,
                            max_task_time,
                            max_iter, 
                            space_step_size,
                            time_step_size ,
                            bias_future_time)
        
        self.rewire_controller = self._get_mpc_controller_for_rewiring()
        self.rewiring_ratio    = rewiring_ratio
        self.kd_tree_future    = KDTree([self.start_node]) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)
        
        if rewiring_radius == -1:
            self.rewiring_radius = 5*self.space_step_size
        else :
            self.rewiring_radius = rewiring_radius


    def _get_mpc_controller_for_rewiring(self) -> TimedMPC:
        """
        Get the MPC controller for the RRT rewiring
        """

        print("Initializing MPC controller for tree rewiring...")
        # Define MPC parameters
        Q = np.eye(self.system.size_state) * 10  # State penalty matrix
        R = np.eye(self.system.size_input) * 1   # Input penalty matrix

        # Create MPC parameters object
        mpc_params = MPCProblem(system  = self.system, 
                                horizon = self.prediction_steps, 
                                Q       = Q, 
                                R       = R, 
                                solver  = "MOSEK")

        # Add input magnitude constraint (elevator angle limited to ±15°)
        mpc_params.add_input_magnitude_constraint(limit = self.max_input_bound, is_hard=True)
        mpc_params.add_general_state_constraints(Hx = self.map.workspace.A, bx = self.map.workspace.b,is_hard=True)
        for constraint in self.stl_constraints :
            mpc_params.add_general_state_time_constraints(Hx = constraint.H, bx = constraint.b, start_time = constraint.start, end_time = constraint.end, is_hard=True)
        
        
        # terminal state constraint 
        mpc_params.reach_refererence_at_steady_state(False) # allows the reference to bereached without being in steady state

        # Create the MPC object
        mpc = TimedMPC(mpc_params)

        return mpc



    def rewire(self) :

        """Rewire the tree with the new node."""
        # Find the neighbors of the new node within a certain radius
        last_node_index = len(self.tree) - 1
        last_node       =  self.tree[last_node_index] # take out the last node

        future_nodes        = [node  if node[self.TIME] > last_node[self.TIME]  else node*1e4  for node in self.tree]
        self.kd_tree_future = KDTree(future_nodes) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)
        nearest_indices     = self.kd_tree_future.query_ball_point(last_node,r= self.rewiring_radius * (np.log(len(self.tree))/len(self.tree))**(1/2))

        for neighbour_index in nearest_indices:

            # do not recheck your own parent
            if neighbour_index == self.parents[last_node_index] :
                continue

            neighbour = self.tree[neighbour_index]

            last_node_state  = last_node[self.STATE]
            last_node_time   = last_node[self.TIME]
            
            neighbour_state  = neighbour[self.STATE]
            neighbour_time   = neighbour[self.TIME]
            

            # Check if the new node is a better parent than the current parent of the neighbour (heuristic)
            expected_speed  = (np.linalg.norm(last_node_state - neighbour_state) + self.cost[last_node_index])/ (self.prediction_steps* self.system.dt + last_node_time)
            
            # attempt rewiring
            if expected_speed < self.cost[neighbour_index]/neighbour_time:
                # Rewire the tree
                try :
                    #  Get trajectory between `from_node` and `to_node` in time `t_max`
                    x_trj, u_trj, new_final_time = self.rewire_controller.get_state_and_control_trajectory(x0 = last_node_state ,t0 = last_node_time, reference = neighbour_state)
                except Exception as e:
                    print(f"Error in rewiring at iteration {self.iteration}, with exception: {e}")
                    continue

                is_in_collision  = False
                new_path_length  = self.cost[last_node_index]

                for ii,x in enumerate(x_trj.T): 
                    for obstacle in self.map.obstacles:
                        if x in obstacle:
                            is_in_collision = True
                            break
                    
                    if ii >=1:
                        new_path_length += np.linalg.norm(x_trj[:,ii] - x_trj[:,ii-1])   

                
                actual_new_speed = new_path_length/(self.prediction_steps* self.system.dt + last_node_time)

                if (not is_in_collision) and (actual_new_speed <= self.cost[neighbour_index]/neighbour_time):
                    
                    self.parents[neighbour_index]      = last_node_index  # the new neighbour is the one that was just added
                    self.trajectories[neighbour_index] = x_trj
                    self.cost[neighbour_index]         = new_path_length



    def plan(self):
        """Run the RRT algorithm to find a path from start to goal with time constraints and barrier function."""
        for iteration in tqdm(range(self.max_iter)):
            self.iteration = iteration
            try:
                self.step()
            except Exception as e:
                print(f"Error in planning at iteration {iteration}, with exception: {e}")
            
            if iteration % self.rewiring_ratio  == 0:
                self.rewire()           

        return self.solutions
