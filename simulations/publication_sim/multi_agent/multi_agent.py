from   matplotlib import pyplot as plt
import numpy as np
np.random.seed(3)

from stl_tool.stl                     import GOp, FOp, TasksOptimizer, ContinuousLinearSystem, BoxBound2d, BoxBound
from stl_tool.environment             import Map
from stl_tool.polytope                import Box2d

from stl_tool.planners import StlRRTStar
from json import loads
import os 


##########################################################
# Create work space and mapo
##########################################################
workspace     = Box2d(x = 0,y = 0,size = 20)**3 # cartesian product of a square 10x10 
map           = Map(workspace = workspace)

# load obstacles 
file_path = os.path.join(os.path.dirname(__file__), "map2d.json")
map_json = loads(open(file_path,mode="r").read())

for object in map_json:
    if object["name"].split("_")[0] == "obstacle":
        center = np.array([object["center_x"], object["center_y"]])
        size   = np.array([object["size_x"], object["size_y"]])
        obstacle = Box2d(x = center[0],y = center[1],size = size)
        obstacle_multi_agent = obstacle**3

        map.add_obstacle(obstacle_multi_agent)

# map.draw() # draw if you want :)
# map.enlarge_obstacle(border_size=0.2) # enlarge obstacles

##########################################################
# system and dynamics
##########################################################
B             = np.diag([1.5,1.5])
dt            = 1.

A_multi = np.random.rand(6,6)*0.0001
B_multi = np.block([[B, np.zeros((2,2)), np.zeros((2,2))],
                    [np.zeros((2,2)), B, np.zeros((2,2))],
                    [np.zeros((2,2)), np.zeros((2,2)), B]])


system             = ContinuousLinearSystem(A_multi, B_multi, dt = dt)
max_input          = 5.
input_bounds       = Box2d(x = 0.,y = 0.,size = max_input*2)**3

##########################################################
# STL specifications
##########################################################

named_map = {item["name"]: item for item in map_json}



# first interest point
intrest_point = named_map["goal1"]
goal1 = BoxBound(dims = [0,1], size = [intrest_point["size_x"],intrest_point["size_y"]] , center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "goal1")
# second interest point
intrest_point = named_map["goal2"]
goal2 = BoxBound(dims = [2,3], size = [intrest_point["size_x"],intrest_point["size_y"]] , center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "goal2")
# third interest point
intrest_point = named_map["goal3"]
goal3 = BoxBound(dims = [3,4], size = [intrest_point["size_x"],intrest_point["size_y"]] , center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "goal3")


gathering_point = BoxBound(dims =[0,1,2,3,4,5], size = 1.3, center= np.array([0., 0., 0., 0., 0., 0.]), name = "gathering_point")

# formula       =  (FOp(130,140) >> goal1)  & (FOp(130,140) >> goal2) & (FOp(130,140) >> goal3) & (GOp(50,60) >> gathering_point)
formula       =  (FOp(130,140) >> goal1)  & (FOp(130,140) >> goal2) & (FOp(130,140) >> goal3) 


# fig,ax = map.draw_formula_predicate(formula = formula, alpha =0.2, projection_dim= [0,1])

##########################################################
# From STL to Barriers
##########################################################
x_01  = np.array([-7.5,7.5])
x_02  = np.array([7.5,-7.5])
x_03  = np.array([8.0,7.5])
x_0   = np.hstack((x_01,x_02,x_03))
# map.show_point(x_0, color = 'r', label = 'start') # show start point

scheduler = TasksOptimizer(formula, workspace,system) # create task optimizer
scheduler.make_time_schedule()                 # create scheduling of the tasks
solver_stats = scheduler.optimize_barriers( input_bounds = input_bounds, x_0 = x_0) # create barriers

# save to file if you want :)
polytope_file = os.path.join(os.path.dirname(__file__), "spec.json")
scheduler.save_polytopes(filename = polytope_file)

# #########################################################
# # Create RRT solver
# #########################################################
# time_varying_constraints = scheduler.get_barrier_as_time_varying_polytopes()
# # scheduler.show_time_varying_level_set(ax,t_start=0.,t_end = 9.99,n_points=20)

# rrt_planner     = StlRRTStar(start_state     = x_0,
#                             system           = system,
#                             prediction_steps = 5,
#                             stl_constraints  = time_varying_constraints ,
#                             map              = map,
#                             max_input        = max_input,
#                             max_iter         = 1000,
#                             space_step_size  = 2.8,
#                             rewiring_radius  = 50,
#                             rewiring_ratio   = 2,
#                             verbose          = True,
#                             biasing_ratio    = 1.2)


# rrt_planner.plan()
# fig,ax = rrt_planner.plot_rrt_solution(ax = ax, solution_only= True)


# ax.scatter(x_0[0], x_0[1], color='r', label='start', s=100)

# rrt_planner.show_statistics()

plt.show()