
import random
import numpy as np
import matplotlib.pyplot as plt
from   typing          import TypedDict
from   tqdm            import tqdm
from   scipy.spatial   import KDTree

from openmpc.mpc     import TimedMPC, MPCProblem
from openmpc.models  import LinearSystem
from openmpc.support import TimedConstraint


from stl_tool.env.map  import Map
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
        

        self.obtained_paths    = []   
        self.new_nodes         = []  
        self.sampled_nodes     = []    
        self.solutions         = []    



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
                                solver  = 'MOSEK')

        # Add input magnitude constraint (elevator angle limited to ±15°)
        mpc_params.add_input_magnitude_constraint(limit = self.max_input_bound, is_hard=True)
        mpc_params.add_general_state_constraints(Hx = self.map.workspace.A, bx = self.map.workspace.b,is_hard=True)
        for constraint in self.stl_constraints :
            mpc_params.add_general_state_time_constraints(Hx = constraint.H, bx = constraint.b, start_time = constraint.start, end_time = constraint.end, is_hard=True)


        # Create the MPC object
        return TimedMPC(mpc_params)



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
        

        from_time  = from_node[2] 
        from_state = from_node[:2]
        to_state   = to_node[:2]

        #  Get trajectory between `from_node` and `to_node` in time `t_max`
        x_trj, u_trj,to_time = self.expand_mpc.get_state_and_control_trajectory(x0 = from_state ,t0 = from_time, reference = to_state)
        
        new_node  = np.hstack((x_trj[:,-1], np.array(to_time)))
        is_in_collision = False
        cost  = 0.
        state = x_trj[:-1,:]
        
        obstacles : list[Polytope] = self.map.obstacles 
        for ii in range(x_trj.shape[1]): 
            for obstacle in obstacles:
                if x_trj[:,ii] in obstacle:
                    is_in_collision = True
                    break
            
            if ii >=1:
                cost += np.linalg.norm(x_trj[:,ii] - x_trj[:,ii-1])   
            
        return new_node, x_trj, is_in_collision, cost

    def plan(self):
        """Run the RRT algorithm to find a path from start to goal with time constraints and barrier function."""
        counter = 0
        for iteration in tqdm(range(self.max_iter)):

            rand_point         = self.random_node()
            past_tree          = [ node if node[2] < rand_point[2] else node*1e6 for node in self.tree ]
            self.kd_tree_past  = KDTree(past_tree) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)

            nearest_node,nearest_index  = self.single_past_nearest(rand_point)
    
            # Move towards the random point with limited direction
            nearest_node_state = nearest_node[:2]
            rand_point_state   = rand_point[:2]
            direction          = rand_point_state - nearest_node_state
            direction          = self.space_step_size * direction / np.linalg.norm(direction)
            rand_point_state   = nearest_node[:2] + direction
            rand_point         = np.hstack((rand_point_state, rand_point[2])) 
            
            try:
                new_node, traj, is_in_collision,cost  = self.steer(nearest_node, rand_point)
            except :
                print(f"Error in steering {counter}")
                counter += 1
                continue

            if not is_in_collision:
                self.tree.append(new_node)
                self.parents.append(nearest_index)
                self.cost.append(self.cost[nearest_index] + cost)
                self.trajectories.append(traj)   


                if new_node[2] >= self.max_task_time and self.cost[-1] < self.current_best_cost:
                    index = len(self.tree) - 1
                    path_traj = []
                    self.current_best_cost = self.cost[-1]
                    
                    while index != -1:
                        path_traj.append(self.trajectories[index])
                        index  = self.parents[index]

                    path_solution = RRTSolution()
                    path_solution["path_trj"]     = path_traj
                    path_solution["cost"]         = cost
                    path_solution["iter"]         = iteration

                    self.solutions.append( path_solution)
                
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
                time = trj[2,:]
                
                if j == len(solution["path_trj"])-1:
                    ax.plot(x, y, lw=4, c = random_color, label="Cost: %.2f"%solution["cost"])
                else :
                    ax.plot(x, y, lw=4, c = random_color)
                # annotate the time at final point 
                ax.annotate(f"t: {time[-1]:.2f}", (x[-1] +0.3 , y[-1]), textcoords="offset points", xytext=(0,10), ha='center')
           
            # find best trajectory in terms of cost
        if len(self.solutions) :
            best = min(self.solutions, key=lambda x: x["cost"])
            for jj,trj in enumerate(best["path_trj"]):
                x    = trj[0,:]
                y    = trj[1,:]
                time = trj[2,:]
                if jj == len(best["path_trj"])-1:
                    ax.plot(x, y, lw=4, c = "red", label="Best Cost: %.2f"%best["cost"])
                else :
                    ax.plot(x, y, lw=4, c = "red")
                # annotate the time at final point 
                ax.annotate(f"t: {time[-1]:.2f}", (x[-1] +0.3 , y[-1]), textcoords="offset points", xytext=(0,10), ha='center', color="red")

            ax.plot(self.start_node[0], self.start_node[1], "go", markersize=10, label="Start")
            ax.legend()
        plt.show()
    



# #########################################################################################################################
# #########################################################################################################################
# class RRTStar(RRT):
#     def __init__(self, step_controller  :LQRcbfControllerLogExp | MPCcbfControllerFreeEnd, 
#                        rewire_controller:MPCcbfControllerFixedEnd,
#                        start_state   :np.ndarray, 
#                        barrier_function :BarrierFunction,
#                        map              :Map,
#                        max_task_time    :float,
#                        max_iter         :int   = 100000, 
#                        space_step_size  :float = 0.3, 
#                        time_step_size   :float = 0.1,
#                        rewiring_radius  :float = -1,
#                        rewiring_ratio  :int   = 2,
#                        bias_future_time :bool = True ) -> None :
        
#         super().__init__(step_controller, start_state, barrier_function, map, max_task_time, max_iter, space_step_size, time_step_size, bias_future_time)
        
#         self.rewire_controller = rewire_controller
#         self.rewire_controller.add_barrier(barrier_function)
#         self.rewire_controller.set_cost_matrices(Q = np.eye(2)*10, R = np.eye(2))
#         self.rewire_controller.setup_solver(horizon_time=self.time_step_size*5)

#         self.rewiring_ratio = rewiring_ratio

#         self.kd_tree_future    = KDTree([self.start_node]) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)
#         if rewiring_radius is -1:
#             self.rewiring_radius = 5*self.space_step_size
#         else :
#             self.rewiring_radius = rewiring_radius

#     def rewire(self) :

#         """Rewire the tree with the new node."""
#         # Find the neighbors of the new node within a certain radius
#         new_node_index = len(self.tree) - 1
#         new_node       =  self.tree[new_node_index ] # take out the last node

#         future_nodes   = [node  if node[2] > new_node[2]  else node*1e4  for node in self.tree]
#         self.kd_tree_future = KDTree(future_nodes) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)
#         nearest_indices = self.kd_tree_future.query_ball_point(new_node,r= self.rewiring_radius * (np.log(len(self.tree))/len(self.tree))**(1/2))

#         for neighbour_index in nearest_indices:

#             # do not recheck your own parent
#             if neighbour_index == self.parents[new_node_index] :
#                 continue

#             neighbour = self.tree[neighbour_index]
#             # Check if the new node is a better parent than the current parent of the neighbour (heuristic)
#             new_cost  = np.linalg.norm(neighbour[:2] - new_node[:2]) + self.cost[new_node_index]
#             # attempt rewiring
#             if new_cost < self.cost[neighbour_index]:

#                 new_node_state = new_node[:2]
#                 neighbour_state = neighbour[:2]
#                 t_0 = new_node[-1]
#                 neighbour_time = neighbour[-1]

#                 # Rewire the tree
#                 try :
#                     traj,u_trj =  self.rewire_controller.bridge(x_0   = new_node_state, 
#                                                                 x_ref = neighbour_state,
#                                                                 t_0   = t_0 ,
#                                                                 t_ref = neighbour_time) # trajectory given as triplets (x,y,t)
#                 except :
#                     continue

#                 is_in_collision  = False
#                 actual_new_cost  = self.cost[new_node_index]
#                 state            = traj[:-1,:]

#                 for ii,x in enumerate(state.T): 

                    
#                     for obstacle in self.map.obstacles:
#                         if (obstacle.x <= x[0] <= (obstacle.x + obstacle.w)) and (obstacle.y <= x[1] <= (obstacle.y + obstacle.h)):
#                             is_in_collision = True
#                             break
                    
#                     if ii >=1:
#                         actual_new_cost += np.linalg.norm(state[:,ii] - state[:,ii-1])   

                
#                 if (not is_in_collision) and (actual_new_cost <= self.cost[neighbour_index]):
                    
                    
                    
#                     self.parents[neighbour_index]      = new_node_index  # the new neighbour is the one that was just added
#                     self.trajectories[neighbour_index] = traj
#                     self.cost[neighbour_index]         = actual_new_cost



#     def plan(self):
#         """Run the RRT algorithm to find a path from start to goal with time constraints and barrier function."""
#         for iteration in tqdm(range(self.max_iter)):

#             rand_point         = self.random_node()
#             past_tree        = [ node if node[2] < rand_point[2] else node*1e6 for node in self.tree ]
#             self.kd_tree_past  = KDTree(past_tree) # KD-tree for nearest neighbour search. Each node is a 3D point (x, y, t)

#             nearest_node,nearest_index  = self.single_past_nearest(rand_point)
    
#             # Move towards the random point with limited direction
#             nearest_node_state = nearest_node[:2]
#             rand_point_state   = rand_point[:2]
#             direction          = rand_point_state - nearest_node_state
#             direction          = self.space_step_size * direction / np.linalg.norm(direction)
#             rand_point_state   = nearest_node[:2] + direction
#             rand_point         = np.hstack((rand_point_state, rand_point[2])) 

#             new_node, traj, is_in_collision,cost  = self.steer(nearest_node, rand_point)


#             if not is_in_collision:
#                 self.tree.append(new_node)
#                 self.parents.append(nearest_index)
#                 self.cost.append(self.cost[nearest_index] + cost)
#                 self.trajectories.append(traj)  

#                 if new_node[2] >= self.max_task_time and self.cost[-1] < self.current_best_cost:
#                     index     = len(self.tree) - 1
#                     path_traj = []
#                     self.current_best_cost = self.cost[-1]
                    
#                     while index != -1:
#                         path_traj.append(self.trajectories[index])
#                         index  = self.parents[index]

#                     path_solution = RRTSolution()
#                     path_solution["path_trj"]     = path_traj
#                     path_solution["cost"]         = self.current_best_cost
#                     path_solution["iter"]         = iteration

#                     self.solutions.append( path_solution)       
            
#             if iteration % self.rewiring_ratio  == 0:
#                 self.rewire()           

#         return self.solutions
