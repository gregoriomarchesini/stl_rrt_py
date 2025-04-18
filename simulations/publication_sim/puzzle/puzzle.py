from   matplotlib import pyplot as plt
import numpy as np

from stl_tool.stl                     import GOp, FOp, TasksOptimizer, BoxBound, ContinuousLinearSystem, ISSDeputy,SingleIntegrator3d, IcosahedronPredicate, ContinuousLinearSystem, BoxBound2d
from stl_tool.environment             import Map,ISSModel
from stl_tool.polytope                import Box2d,Box3d,Icosahedron

from stl_tool.planners import RRTStar,RRT, BiasedSampler
from copy import copy
from json import loads
import os 
##########################################################
# Create work space and mapo
##########################################################
workspace     = Box2d(x = 0,y = 0,size = 20) # square 20x20
map           = Map(workspace = workspace)

# load obstacles 
file_path = os.path.join(os.path.dirname(__file__), "map2d.json")

with open(file_path, "r") as f:
    map_json = loads(f.read())

obstacles = []
for object in map_json:
    if object["name"].split("_")[0] == "obstacle":
        center = np.array([object["center_x"], object["center_y"]])
        size   = np.array([object["size_x"], object["size_y"]])
        if len(center) == 2 and len(size) == 2:
            obstacles.append(Box2d(x = center[0],y = center[1],size = size))
        else:
            raise ValueError("Obstacle size and center must be 2D vectors")


map.add_obstacle(obstacles)


map.draw() # draw if you want :)

##########################################################
# system and dynamics
##########################################################
A             = np.eye(2)*0.
B             = np.eye(2)
dt = 0.1
system        = ContinuousLinearSystem(A, B, dt = dt)
max_input     = 4.8
input_bounds  = Box2d(x = 0.,y = 0.,size = max_input*2) 


##########################################################
# STL specifications
##########################################################

named_map = {item["name"]: item for item in map_json}

# first interest point
intrest_point = named_map["interest_2"]
p1 = BoxBound2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "interest_1")
# second interest point
intrest_point = named_map["interest_6"]
p2 = BoxBound2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "interest_2")
# third interest point
intrest_point = named_map["interest_3"]
p3 = BoxBound2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "interest_3")

intrest_point = named_map["interest_4"]
p4 = BoxBound2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "interest_3")

# charginig_station 
intrest_point = named_map["charging_station"]
c_station = BoxBound2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "charging_station")

formula   =  (FOp(30,40) >> p1)  & (FOp(70,80) >> p2) 

# (GOp(0.01,80) >>  (FOp(0.01,60) >> c_station)) &
# & (FOp(13,20) >> p2)  & (FOp(16,20) >> p3)  & FOp(16,20) >> p4 & (GOp(0,200) >>  (FOp(0,60) >> c_station))




fig,ax = map.draw_formula_predicate(formula = formula)
# # formula.show_graph()

# # ##########################################################
# # # From STL to Barriers
# # ##########################################################
x_0       = c_station.polytope.sample_random()
map.show_point(x_0, color = 'r', label = 'start') # show start point

scheduler = TasksOptimizer(formula, workspace,system) # create task optimizer
scheduler.make_time_schedule()                 # create scheduling of the tasks
# scheduler.plot_time_schedule()               # visualize distribution of tasks time durations :)
solver_stats = scheduler.optimize_barriers( input_bounds = input_bounds,     # optimize barrier functions
                                            x_0          = x_0)

# save to file if you want :)
scheduler.save_polytopes(filename= "test_polytopes")

#########################################################
# Create RRT solver
#########################################################
time_varying_constraints = scheduler.get_barrier_as_time_varying_polytopes()
# scheduler.show_time_varying_level_set()

list_of_beta_polytope_pairs = scheduler.get_list_of_beta_polytopes_pairs()

list_of_polytopes = [polytope for beta,polytope in list_of_beta_polytope_pairs]
list_of_betas     = [beta     for beta,polytope in list_of_beta_polytope_pairs]

sampler = BiasedSampler(list_of_polytopes = list_of_polytopes,list_of_times = list_of_betas)



prediction_steps   = 10
space_step         = max_input/4 * dt * prediction_steps

rrt_planner        = RRTStar(start_state     = x_0,
                            system           = system,
                            prediction_steps = 6,
                            stl_constraints  = time_varying_constraints ,
                            map              = map,
                            max_input        = max_input,
                            max_task_time    = formula.max_horizon(),
                            max_iter         = 2000,
                            space_step_size  = 4,
                            rewiring_radius  = 100,
                            rewiring_ratio   = 1)

# rrt_planner        = RRT(start_state    = x_0,
#                         system           = system,
#                         prediction_steps = 6,
#                         stl_constraints  = time_varying_constraints ,
#                         map              = map,
#                         max_input        = max_input,
#                         max_task_time    = formula.max_horizon(),
#                         max_iter         = 5000,
#                         space_step_size  = 4,
#                         sampler          = sampler,
#                         verbose= True,)


rrt_planner.plan()
fig,ax = rrt_planner.plot_rrt_solution(ax = ax)



ax.scatter(x_0[0], x_0[1], color='r', label='start', s=100)
plt.title("Box Predicate")

rrt_planner.show_statistics()


plt.show()

plt.show()