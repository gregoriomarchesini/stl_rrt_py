from   matplotlib import pyplot as plt
import numpy as np

from stl_tool.stl               import GOp, FOp, TasksOptimizer, BoxBound, ContinuousLinearSystem
from stl_tool.environment.map   import Map
from stl_tool.polytope          import Box2d, BoxNd
from stl_tool.planners          import RRT     


##########################################################
# Create work space and mapo
##########################################################
workspace     = BoxNd(n_dim=4 ,size =[15,15,2,2])
map           = Map(workspace = workspace)

# create obstacles 
map.add_obstacle(Box2d(x = 3,y = 3,size = 1).get_inflated_polytope_to_dimension(dim=4))
map.add_obstacle(Box2d(x = 4,y = -4,size = 1).get_inflated_polytope_to_dimension(dim=4))
map.add_obstacle(Box2d(x = 5,y = -1,size = 1).get_inflated_polytope_to_dimension(dim=4))
map.add_obstacle(Box2d(x = 0,y = -2,size = 1.5).get_inflated_polytope_to_dimension(dim=4))

# map.draw() # draw if you want :)

##########################################################
# system and dynamics
##########################################################
A             = np.block([[np.zeros((2,2)),np.eye(2)],
                         [np.zeros((2,2)),np.zeros((2,2))]])
B             = np.vstack((np.zeros((2,2)),np.eye(2)))
C             = np.hstack((np.eye(2),np.zeros((2,2)))) 


system        = ContinuousLinearSystem(A = A, B = B, C= C)
max_input     = 3.4
input_bounds  = Box2d(x = 0.,y = 0.,size = max_input*2) 


##########################################################
# STL specifications
##########################################################
center         = np.array([-0., 0.])
box_predicate  = BoxBound(dims=[0,1], size = 2.8, center = center)
box2_predicate = BoxBound(dims=[0,1], size = 2.8, center = center + np.array([-3,-3.]))
box3_predicate = BoxBound(dims=[0,1], size = 2.8, center = center + np.array([3.,3.]))
formula        = (GOp(10.,14.) >> box_predicate)  & (FOp(17.,20.) >> box2_predicate) & (FOp(29.,30.) >> box3_predicate)
map.draw_formula_predicate(formula = formula,projection_dim=[0,1])
formula.show_graph()

#########################################################
# From STL to Barriers
#########################################################
x_0       = np.hstack((center + np.array([2,-2.]),np.zeros(2)))
map.show_point(x_0, color = 'r', label = 'start') # show start point

scheduler = TasksOptimizer(formula = formula , workspace = workspace, system = system) # create task optimizer
scheduler.make_time_schedule()                 # create scheduling of the tasks
# scheduler.plot_time_schedule()               # visualize distribution of tasks time durations :)
solver_stats = scheduler.optimize_barriers( input_bounds = input_bounds,     # optimize barrier functions
                                            x_0          = x_0)

# save to file if you want :)
scheduler.save_polytopes(filename= "test_polytopes")

########################################################
# Create RRT solver
########################################################
time_varying_constraints = scheduler.get_barrier_as_time_varying_polytopes()
# scheduler.show_time_varying_level_set()

# 3d axis
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim([-30, 30])
ax.set_ylim([-30, 30])
ax.set_zlim([-30, 30])


ax.scatter(x_0[0], x_0[1], 0, color='r', label='start')



rrt_planner = RRT(start_state      = x_0,
                  system           = system,
                  prediction_steps = 10,
                  stl_constraints  = time_varying_constraints ,
                  map              = map,
                  max_input        = 1000,
                  max_task_time    = formula.max_horizon(),
                  max_iter         = 1000,)


rrt_planner.plan()
fig,ax = rrt_planner.plot_rrt_solution()
ax.scatter(x_0[0], x_0[1], color='r', label='start', s=100)
plt.title("Box Predicate")
plt.show()