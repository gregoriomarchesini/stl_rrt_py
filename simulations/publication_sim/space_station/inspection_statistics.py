from   matplotlib import pyplot as plt
import numpy as np
np.random.seed(3)

from stl_tool.stl                     import (GOp, 
                                              FOp, 
                                              BoxPredicate, 
                                              ISSDeputy,
                                              compute_polyhedral_constraints,
                                              TimeVaryingConstraint)
from stl_tool.environment             import Map,ISSModel
from stl_tool.polyhedron              import Box3d,Icosahedron

from stl_tool.planners import StlRRTStar
from copy import copy


##########################################################
# Create work space and mapo
##########################################################
position_box_halfsize  = 130
velocity_box_halfsize  = 0.25
position_workspace = Box3d(x = 0,y = 0, z= 0, size = 2*position_box_halfsize) 
velocity_workspace = Box3d(x = 0,y = 0, z= 0, size = 2*velocity_box_halfsize)

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
box_size = 70
interest_point_1_center  = np.array([-100., 100., 0.])
box_predicate_1          =  BoxPredicate(dims = [0,1,2], size = box_size, center = interest_point_1_center)
visit_time1              = 1000.

interest_point_2_center  = np.array([-100., -100., 0.])
box_predicate_2          =  BoxPredicate(dims = [0,1,2], size = box_size, center = interest_point_2_center)
visit_time2               = 2500.

interest_point_4_center  = np.array([0., 0., 100.])
box_predicate_4          =  BoxPredicate(dims = [0,1,2], size = box_size, center = interest_point_4_center)
visit_time4              = 3500.

interest_point_3_center  = np.array([100., 100., 0.])
box_predicate_3          =  BoxPredicate(dims = [0,1,2], size = box_size, center = interest_point_3_center)
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

time_varying_constraints,robustness  = compute_polyhedral_constraints(formula      =  formula,
                                                                      workspace    = workspace, 
                                                                      system       = system,
                                                                      input_bounds = input_bounds,
                                                                      x_0          = x_0,
                                                                      plot_results = True,
                                                                      kappa_gain       = 0.070,)

##########################################################
# Plot time varying sets
##########################################################
# print(len(time_varying_constraints), "time varying constraints")
# for tvc in time_varying_constraints:
#     tvc.plot3d(ax = ax, alpha =0.01,color = 'g')



first_sol_time : list[float]= []
first_sol_cost : list[float]= []
best_sol_time  : list[float]= []
best_sol_cost  : list[float]= []
failures       : int = 0


for jj in range(100):
    rrt_planner     = StlRRTStar(start_state      = x_0,
                                 system           = system,
                                 prediction_steps = 5,
                                 stl_constraints  = time_varying_constraints ,
                                 map              = map,
                                 max_input        = max_input,
                                 max_iter         = 1000,
                                 space_step_size  = 15,
                                 rewiring_radius  = 25,
                                 rewiring_ratio   = 5,
                                 biasing_ratio    = 2.)



    solution, stats = rrt_planner.plan()
    # fig,ax = rrt_planner.plot_rrt_solution(ax = ax, solution_only= False)

    if stats["first_sol_clock_time"] is not None :
        first_sol_time.append(stats["first_sol_clock_time"])
        first_sol_cost.append(stats["first_sol_cost"])
        best_sol_time.append(stats["best_sol_clock_time"])
        best_sol_cost.append(stats["best_sol_cost"])
    else:
        failures += 1
        print("no solution found")


print("--------------------------------------------------")
print("Average first solution time: ", np.mean(first_sol_time))
print("Average first solution cost: ", np.mean(first_sol_cost))
print("standard deviation of first solution time: ", np.std(first_sol_time))
print('standard deviation of first solution cost: ', np.std(first_sol_cost))


print("Average best solution time: ", np.mean(best_sol_time))
print("Average best solution cost: ", np.mean(best_sol_cost))
print("standard deviation of best solution time: ", np.std(best_sol_time))
print('standard deviation of best solution cost: ', np.std(best_sol_cost))
print("--------------------------------------------------")

fig,ax = plt.subplots(1,2, figsize=(10,10))
# show clock tand cost with standard deviation bound
ax[0].plot(first_sol_time, label = "first solution time")
ax[0].plot(best_sol_time, label = "best solution time")
ax[0].set_title("clock time")
ax[0].set_xlabel("iteration")
ax[0].set_ylabel("time")
ax[0].legend()
ax[1].plot(first_sol_cost, label = "first solution cost")
ax[1].plot(best_sol_cost, label = "best solution cost")
ax[1].set_title("cost")
ax[1].set_xlabel("iteration")
ax[1].set_ylabel("cost")
ax[1].legend()


# save result to txt
import os
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
filename = "statistics_" + current_time + ".txt"
file_folder = os.path.dirname(__file__)
if not os.path.exists("statistics"):
    os.makedirs(file_folder + "/statistics", exist_ok=True)
with open(file_folder + "/statistics/" + filename, "w") as f:
    f.write("Average first solution time: " + str(np.mean(first_sol_time)) + "\n")
    f.write("Average first solution cost: " + str(np.mean(first_sol_cost)) + "\n")
    f.write("standard deviation of first solution time: " + str(np.std(first_sol_time)) + "\n")
    f.write('standard deviation of first solution cost: ' + str(np.std(first_sol_cost)) + "\n")
    f.write("Average best solution time: " + str(np.mean(best_sol_time)) + "\n")
    f.write("Average best solution cost: " + str(np.mean(best_sol_cost)) + "\n")
    f.write("standard deviation of best solution time: " + str(np.std(best_sol_time)) + "\n")
    f.write('standard deviation of best solution cost: ' + str(np.std(best_sol_cost)) + "\n")


# save time series as numpy array
np.save(file_folder + "/statistics/first_sol_time.npy", first_sol_time)
np.save(file_folder + "/statistics/first_sol_cost.npy", first_sol_cost)
np.save(file_folder + "/statistics/best_sol_time.npy", best_sol_time)
np.save(file_folder + "/statistics/best_sol_cost.npy", best_sol_cost)
    
rrt_planner.show_statistics()
plt.show()