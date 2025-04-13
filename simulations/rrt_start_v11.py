from   matplotlib import pyplot as plt
import numpy as np
from   openmpc import LinearSystem
from   openmpc.support import TimedConstraint

from stl_tool.stl                     import GOp, FOp
from stl_tool.stl.parameter_optimizer import TasksOptimizer
from stl_tool.stl.predicate_models    import BoxPredicate
from stl_tool.environment.map         import Map
from stl_tool.polytope                import Box2d

from stl_tool.planners import RRT, RRTStar




##########################################################
# Create work space and mapo
##########################################################
workspace     = Box2d(x = 0,y = 0,size = 15)
map           = Map(workspace = workspace)

# create obstacles 
map.add_obstacle(Box2d(x = 3,y = 3,size = 1))
map.add_obstacle(Box2d(x = 4,y = -4,size = 1))
map.add_obstacle(Box2d(x = 5,y = -1,size = 1))
map.add_obstacle(Box2d(x = 0,y = -2,size = 1.5))

# map.draw() # draw if you want :)

##########################################################
# system and dynamics
##########################################################
A             = np.eye(2)*0.
B             = np.eye(2)
system        = LinearSystem.c2d(A, B, dt = 0.1)
max_input     = 3.4
input_bounds  = Box2d(x = 0.,y = 0.,size = max_input*2) 


##########################################################
# STL specifications
##########################################################
center         = np.array([-0., 0.])
box_predicate  = BoxPredicate(n_dim = 2, size = 2.8, center = center)
box2_predicate = BoxPredicate(n_dim = 2, size = 2.8, center = center + np.array([-3,0.]))
box3_predicate = BoxPredicate(n_dim = 2, size = 2.8, center = center + np.array([3,0.]))
formula        = (GOp(10.,14.) >> box_predicate)  & (FOp(17.,20.) >> box2_predicate) & (FOp(29.,30.) >> box3_predicate) 
map.draw_formula_predicate(formula = formula)
formula.show_graph()

##########################################################
# From STL to Barriers
##########################################################
x_0       = center + np.array([2,-2.])
# map.show_point(x_0, color = 'r', label = 'start') # show start point

scheduler = TasksOptimizer(formula, workspace) # create task optimizer
scheduler.make_time_schedule()                 # create scheduling of the tasks
# scheduler.plot_time_schedule()               # visualize distribution of tasks time durations :)
solver_stats = scheduler.optimize_barriers( input_bounds = input_bounds,     # optimize barrier functions
                                            system       = system,
                                            x_0          = x_0)

# # save to file if you want :)
# scheduler.save_polytopes(filename= "test_polytopes")

#########################################################
# Create RRT solver
#########################################################
time_varying_constraints = scheduler.get_barrier_as_time_varying_polytopes()
# scheduler.show_time_varying_level_set()

# 3d axis
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim([-30, 30])
ax.set_ylim([-30, 30])
ax.set_zlim([-30, 30])


ax.scatter(x_0[0], x_0[1], 0, color='r', label='start')


rrt_constraints          = []
for tvc in time_varying_constraints:
    rrt_constraints.append(TimedConstraint(H     = tvc.H,
                                           b     = tvc.b, 
                                           start = tvc.start_time,
                                           end   = tvc.end_time))



rrt_planner = RRTStar(start_state      = x_0,
                        system           = system,
                        prediction_steps = 10,
                        stl_constraints  = rrt_constraints,
                        map              = map,
                        max_input        = 1000,
                        max_task_time    = formula.max_horizon(),
                        max_iter         = 1000,
                        bias_future_time = False,
                        rewiring_radius  = 10)


rrt_planner.plan()
fig,ax = rrt_planner.plot_rrt_solution()
ax.scatter(x_0[0], x_0[1], color='r', label='start', s=100)
plt.title("Box Predicate")
plt.show()