from   matplotlib import pyplot as plt
import numpy as np
np.random.seed(3)

from stl_tool.stl                     import GOp, FOp, TasksOptimizer, ContinuousLinearSystem, BoxBound2d, RegularPolygonPredicate2D
from stl_tool.environment             import Map
from stl_tool.polyhedron                import Box2d,RegularPolygon2D

from stl_tool.planners import StlRRTStar
from json import loads
import os 


##########################################################
# Create work space and mapo
##########################################################
n_sides = 4
radius_ws = 10
workspace     = RegularPolygon2D(n_sides = n_sides,
                                 size    = radius_ws,
                                 center  = np.array([0.,0.]),
                                 name    = "workspace")
map           = Map(workspace = workspace)

# # load obstacles 
# file_path = os.path.join(os.path.dirname(__file__), "map2d.json")
# map_json = loads(open(file_path,mode="r").read())
# map.read_from_json(file_path)
map.draw() # draw if you want :)
# map.enlarge_obstacle(border_size=0.2) # enlarge obstacles

##########################################################
# system and dynamics
##########################################################
A             = np.random.rand(2,2)*0.1
B             = np.diag([1.5,1.5])
dt            = 1.
system        = ContinuousLinearSystem(A, B, dt = dt)
max_input     = 1.
input_bounds  = Box2d(x = 0.,y = 0.,size = max_input*2) 

print(A)
##########################################################
# STL specifications
##########################################################


radius = 2.5
p4     = RegularPolygonPredicate2D(n_sides= n_sides,
                                    size   = radius,
                                    center = np.array([0., 0.]),
                                    name   = "interest_4")

formula       =  GOp(10,20) >> p4 


fig,ax = map.draw_formula_predicate(formula = formula, alpha =0.2)

# # ##########################################################
# # # From STL to Barriers
# # ##########################################################
x_0       = np.array([5., 5.]) # start point
map.show_point(x_0, color = 'r', label = 'start') # show start point

scheduler = TasksOptimizer(formula, workspace,system) # create task optimizer
scheduler.make_time_schedule()                 # create scheduling of the tasks
solver_stats = scheduler.optimize_barriers( input_bounds = input_bounds, x_0 = x_0) # create barriers

# save to file if you want :)
polytope_file = os.path.join(os.path.dirname(__file__), "spec.json")
scheduler.save_polytopes(filename = polytope_file)

#########################################################
# Create RRT solver
#########################################################
time_varying_constraints = scheduler.get_barrier_as_time_varying_polytopes()
# scheduler.show_time_varying_level_set(ax,t_start=0.,t_end = 9.99,n_points=20)

rrt_planner     = StlRRTStar(start_state     = x_0,
                            system           = system,
                            prediction_steps = 5,
                            stl_constraints  = time_varying_constraints ,
                            map              = map,
                            max_input        = max_input,
                            max_iter         = 500,
                            space_step_size  = 2.8,
                            rewiring_radius  = 25,
                            rewiring_ratio   = 2,
                            verbose          = True,
                            biasing_ratio    = 2)


rrt_planner.plan()
fig,ax = rrt_planner.plot_rrt_solution(ax = ax, solution_only= False)


ax.scatter(x_0[0], x_0[1], color='r', label='start', s=100)

rrt_planner.show_statistics()

plt.show()