from   matplotlib import pyplot as plt
import numpy as np
import os
np.random.seed(3)

from stl_tool.stl                     import (GOp, 
                                              FOp, 
                                              ContinuousLinearSystem, 
                                              BoxPredicate2d,
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
map.read_obstacles_from_json(file_path)
map.draw() # draw if you want :)
map.enlarge_obstacle(border_size=0.2) # enlarge obstacles

##########################################################
# system and dynamics
##########################################################
A             = (np.random.rand(2,2)-1.)*0.1
B             = np.diag([1.,1.])
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
p1 = BoxPredicate2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "interest_1")
# second interest point
intrest_point = named_map["interest_6"]
p2 = BoxPredicate2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "interest_2")
# third interest point
intrest_point = named_map["interest_3"]
p3 = BoxPredicate2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "interest_3")

intrest_point = named_map["interest_4"]
p4 = BoxPredicate2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "interest_3")

# charging_station 
intrest_point = named_map["charging_station"]
c_station     = BoxPredicate2d(size = intrest_point["size_x"], center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "charging_station")

formula1       =  (FOp(20,25) >> p1)  & (FOp(150,155) >> p2) & (GOp(0.01,200) >>  (FOp(0,140) >> c_station)) & (GOp(260,265) >> p3)
formula2       =  (FOp(20,25) >> p2)  & (FOp(150,155) >> p3) & (GOp(0.01,200) >>  (FOp(0,140) >> c_station)) & (GOp(260,265) >> p1)

# formula       =  (FOp(20,25) >> p1)  & (FOp(150,155) >> p2)
fig,ax = map.draw_formula_predicate(formula = formula1, alpha =0.2)

# # ##########################################################
# # # From STL to Barriers
# # ##########################################################
x_0       = c_station.polytope.sample_random()
map.show_point(x_0, color = 'r', label = 'start') # show start point


time_varying_constraints1,robustness_1,kappa_gain_1   = compute_polyhedral_constraints(formula      =  formula1,
                                                                          workspace    = workspace, 
                                                                          system       = system,
                                                                          input_bounds = input_bounds,
                                                                          x_0          = x_0,
                                                                          plot_results = True)

time_varying_constraints2, robustness_2, kappa_gain_2 = compute_polyhedral_constraints(formula      =  formula2,
                                                                         workspace    = workspace, 
                                                                         system       = system,
                                                                         input_bounds = input_bounds,
                                                                         x_0          = x_0,
                                                                         plot_results = True)


if robustness_2 >= robustness_1:
    time_varying_constraints = time_varying_constraints2
    print("Using formula 1")
else:
    time_varying_constraints = time_varying_constraints1
    print("Using formula 2")


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

first_sol_time : list[float]= []
first_sol_cost : list[float]= []
best_sol_time  : list[float]= []
best_sol_cost  : list[float]= []
failures       : int = 0


for jj in range(100):
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