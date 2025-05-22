from   matplotlib import pyplot as plt
import numpy as np
import os
np.random.seed(3)

from stl_tool.stl                     import (GOp, 
                                              FOp, 
                                              ContinuousLinearSystem, 
                                              BoxBound2d,
                                              compute_polyhedral_constraints,
                                              TimeVaryingConstraint)
from stl_tool.environment             import Map
from stl_tool.polyhedron                import Box2d

from stl_tool.planners import StlRRTStar
from json import loads
import os 


##########################################################
# Create work space and mapo
##########################################################
workspace     = Box2d(x = 0,y = 0,size = 20) # square 20x20
map           = Map(workspace = workspace)

# load obstacles 
file_path = os.path.join(os.path.dirname(__file__), "map2d.json")
map_json = loads(open(file_path,mode="r").read())
map.read_from_json(file_path)
map.draw() # draw if you want :)
map.enlarge_obstacle(border_size=0.2) # enlarge obstacles

##########################################################
# system and dynamics
##########################################################
A             = np.random.rand(2,2)*0.1
B             = np.diag([1.5,1.5])
dt            = 1.0
system        = ContinuousLinearSystem(A, B, dt = dt)
max_input     = 5.0
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

# charging_station 
intrest_point = named_map["charging_station"]
c_station     = BoxBound2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "charging_station")

formula       =  (FOp(20,25) >> p1)  & (FOp(150,155) >> p2) & (GOp(0.01,200) >>  (FOp(0,140) >> c_station)) & (GOp(260,265) >> p3)

# formula       =  (FOp(20,25) >> p1)  & (FOp(150,155) >> p2)
fig,ax = map.draw_formula_predicate(formula = formula, alpha =0.2)

# # ##########################################################
# # # From STL to Barriers
# # ##########################################################
x_0       = c_station.polytope.sample_random()
map.show_point(x_0, color = 'r', label = 'start') # show start point


time_varying_constraints : list[TimeVaryingConstraint] = compute_polyhedral_constraints(formula      =  formula,
                                                                                        workspace    = workspace, 
                                                                                        system       = system,
                                                                                        input_bounds = input_bounds,
                                                                                        x_0          = x_0,
                                                                                        plot_results = True)




# # for tvc in time_varying_constraints:
# #     try :
# #         tvc.plot2d()
# #         print("start_time",tvc.start_time)
# #         print("end_time",tvc.end_time)
# #     except :
# #         print("problem with plotting the constraint")
# #         poly = tvc.to_polytope()
# #         print(poly.is_open)
# #         continue

# ########################################################
# # Create RRT solver
# ########################################################





rrt_planner     = StlRRTStar(start_state     = x_0,
                            system           = system,
                            prediction_steps = 5,
                            stl_constraints  = time_varying_constraints ,
                            map              = map,
                            max_input        = max_input,
                            max_iter         = 700,
                            space_step_size  = 2.8,
                            rewiring_radius  = 25,
                            rewiring_ratio   = 2,
                            verbose          = True,
                            biasing_ratio    = 1.5,
                            )


solution, stats = rrt_planner.plan()
fig,ax = rrt_planner.plot_rrt_solution(ax = ax, solution_only= False)

rrt_planner.show_statistics()

plt.show()