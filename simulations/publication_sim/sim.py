from   matplotlib import pyplot as plt
import numpy as np

from stl_tool.stl                     import GOp, FOp, TasksOptimizer, BoxBound, ContinuousLinearSystem, ISSDeputy
from stl_tool.environment             import Map,ISSModel
from stl_tool.polytope                import Box2d,Box3d,Icosahedron

from stl_tool.planners import RRTStar


##########################################################
# Create work space and mapo
##########################################################
workspace     = Box3d(x = 0,y = 0, z= 0, size = 150) + Box3d(x = 0,y = 0, z= 0, size = 0.4) # cartesian product for position and velocity box
map           = Map(workspace = workspace)

# create obstacles 
map.add_obstacle(Icosahedron(radius=75,x = 0,y=0,z=0))


iss = ISSModel()
fig,ax = iss.plot(elev=30, azim=45)
map.draw(ax) # draw if you want :)

##########################################################
# system and dynamics
##########################################################
system        = ISSDeputy(dt = 0.1,r0=7000e3) # Example: r0 = 7000 km
max_input     = 25.
input_bounds  = Box3d(x = 0.,y = 0.,z=0.,size = max_input*2) 


##########################################################
# STL specifications
##########################################################
interest_point_1_center  = np.array([-80., 80., 0.])
box_predicate_1          =  BoxBound(dims = [0,1,2], size = 70, center = interest_point_1_center)

interest_point_2_center  = np.array([-80., -80., 0.])
box_predicate_2          =  BoxBound(dims = [0,1,2], size = 70, center = interest_point_2_center)

interest_point_3_center  = np.array([80., 80., 0.])
box_predicate_3          =  BoxBound(dims = [0,1,2], size = 70, center = interest_point_3_center)



formula        = (GOp(40.,45.) >> box_predicate_1)  & (FOp(100.,120.) >> box_predicate_2) & (FOp(180.,185.) >> box_predicate_3) 
fig,ax = map.draw_formula_predicate(formula = formula)
# formula.show_graph()

##########################################################
# From STL to Barriers
##########################################################
x_0       = np.array([-100., 0., 0., 0.001,0.01,0.001]) # initial state
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

print(time_varying_constraints[0].H)
print(time_varying_constraints[1].H)




# rrt_planner = RRTStar(start_state      = x_0,
#                       system           = system,
#                       prediction_steps = 10,
#                       stl_constraints  = time_varying_constraints ,
#                       map              = map,
#                       max_input        = 1000,
#                       max_task_time    = formula.max_horizon(),
#                       max_iter         = 1000,
#                       bias_future_time = False)


# rrt_planner.plan()
# fig,ax = rrt_planner.plot_rrt_solution(ax = ax)

# ax.scatter(x_0[0], x_0[1], color='r', label='start', s=100)
# plt.title("Box Predicate")


plt.show()




