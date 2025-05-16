from   matplotlib import pyplot as plt
import numpy as np
np.random.seed(3)

from stl_tool.stl                     import GOp, FOp, TasksOptimizer, BoxBound, ContinuousLinearSystem, ISSDeputy,SingleIntegrator3d
from stl_tool.environment             import Map,ISSModel
from stl_tool.polyhedron                import Box2d,Box3d,Icosahedron

from stl_tool.planners import StlRRTStar
from copy import copy


##########################################################
# Create work space and mapo
##########################################################
position_workspace = Box3d(x = 0,y = 0, z= 0, size = 2*130) 
velocity_workspace = Box3d(x = 0,y = 0, z= 0, size = 0.5)

workspace     = position_workspace * velocity_workspace # cartesian product of the  two boxes
map           = Map(workspace = workspace)

# create obstacles 
map.add_obstacle(Icosahedron(radius=75,x = 0,y=0,z=0)) # add obstacle in poistion around the ISS


iss = ISSModel()
fig,ax = iss.plot(elev=48, azim=143)
map.draw(ax, alpha = 0.1) # draw if you want :)

##########################################################
# system and dynamics
##########################################################
system        = ISSDeputy(dt = 5, )
max_input     = 1.5
input_bounds  = Box3d(x = 0.,y = 0.,z=0.,size = max_input*2) 


##########################################################
# STL specifications
##########################################################
interest_point_1_center  = np.array([-100., 100., 0.])
box_predicate_1          =  BoxBound(dims = [0,1,2], size = 70, center = interest_point_1_center)
visit_time1              = 1000.

interest_point_2_center  = np.array([-100., -100., 0.])
box_predicate_2          =  BoxBound(dims = [0,1,2], size = 70, center = interest_point_2_center)
visit_time2               = 2500.

interest_point_4_center  = np.array([0., 0., 100.])
box_predicate_4          =  BoxBound(dims = [0,1,2], size = 70, center = interest_point_4_center)
visit_time4              = 3500.

interest_point_3_center  = np.array([100., 100., 0.])
box_predicate_3          =  BoxBound(dims = [0,1,2], size = 70, center = interest_point_3_center)
visit_time3              = 5000.

visit_period             = 400


formula        = ((FOp(visit_time1,visit_time1+ visit_period/4) >> (GOp(0, visit_period) >> box_predicate_1))  & 
                  (FOp(visit_time2,visit_time2+ visit_period/4) >> (GOp(0, visit_period) >> box_predicate_2))  &  
                  (FOp(visit_time4,visit_time4+ visit_period/4) >> (GOp(0, visit_period) >> box_predicate_4))  & 
                  (FOp(visit_time3,visit_time3+ visit_period/4) >> (GOp(0, visit_period) >> box_predicate_3)) )
 
fig,ax = map.draw_formula_predicate(formula = formula, alpha = 0.2)
# formula.show_graph()

##########################################################
# From STL to Barriers
##########################################################
x_0       = np.array([-100., 0., -50., 0.,0. , 0.]) # initial state
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


rrt_planner     = StlRRTStar(start_state     = x_0,
                            system           = system,
                            prediction_steps = 5,
                            stl_constraints  = time_varying_constraints ,
                            map              = map,
                            max_input        = max_input,
                            max_iter         = 500,
                            space_step_size  = 5,
                            rewiring_radius  = 25,
                            rewiring_ratio   = 5,
                            biasing_ratio    = 2)



rrt_planner.plan()
fig,ax = rrt_planner.plot_rrt_solution(ax = ax, solution_only=True)
ax.view_init(elev=48, azim=143)

rrt_planner.show_statistics()


plt.show()