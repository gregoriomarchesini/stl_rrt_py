from   matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import os 

from stl_tool.stl                     import (GOp, 
                                              FOp, 
                                              SingleIntegrator2d,
                                              MultiAgentSystem,
                                              RelativeFormationTuple,
                                              compute_polyhedral_constraints)

from stl_tool.stl.logic     import Formula
from stl_tool.controllers   import MPCProblem, TimedMPC
from stl_tool.controllers   import AnalyticalQp
"""
Design of time-varying sets for mas of single integrators
"""
##########################################################
# Create work space 
##########################################################
input_bounds = 3.5
num_agents   = 4
systems      = []
for agents in range(num_agents):
    system = SingleIntegrator2d(name = f"agent_{agents+1}", dt = 0.2)
    system.set_workspace_bounds(ubx = [10., 10.],lbx = [-10., -10.])
    system.set_input_bounds(ubu = [input_bounds, input_bounds],lbu = [-input_bounds, -input_bounds])
    systems.append(system)
    system.print_states_names()

mas = MultiAgentSystem(systems = systems)
print("The workspace of the multi-agent system is:")
print(mas.workspace)
print("Input bounds of each agent are:")
print(mas.inputbounds)


##########################################################
# STL specifications
##########################################################

# Define formation specification
formation_1 = RelativeFormationTuple( system_1      = "agent_1",
                                      system_2      = "agent_2",
                                      center        = np.array([5.0, 0.0]),
                                      size          = np.array([1.5, 1.5]),
                                      state_name_1  = "position",
                                      state_name_2  = "position")

formation_2 = RelativeFormationTuple( system_1      = "agent_1",
                                      system_2      = "agent_3",
                                      center        = np.array([-5.0, 0.0]),
                                      size          = np.array([1.5, 1.5]),
                                      state_name_1  = "position",
                                      state_name_2  = "position")

formation_3 = RelativeFormationTuple( system_1      = "agent_1",
                                      system_2      = "agent_4",
                                      center        = np.array([0.0, -5.0]),
                                      size          = np.array([1.5, 1.5]),
                                      state_name_1  = "position",
                                      state_name_2  = "position")

formation_predicate = mas.get_relative_formation_box_predicate(formation_1, formation_2, formation_3)
agent_1_predicate   = mas.get_single_agent_box_predicate_by_name( system_name = "agent_1",
                                                                  center      = np.array([0.0, 0.0]),
                                                                  size        = [1.0, 1.0],
                                                                  state_name  = "position")

agent_3_predicate   = mas.get_single_agent_box_predicate_by_name( system_name = "agent_3",
                                                                  center      = np.array([0.0, 0.0]),
                                                                  size        = [1.0, 1.0],
                                                                  state_name  = "position")

agent_2_predicate   = mas.get_single_agent_box_predicate_by_name( system_name = "agent_2",
                                                                  center      = np.array([0.0, 0.0]),
                                                                  size        = [1.0, 1.0],
                                                                  state_name  = "position")



x_0 = np.array([ -10.0,  8.0,   # agent 1 position
                  8.0,  6.0,   # agent 2 position
                7.0,  4.0,
                6.0,  2.0])  # agent 3 position

print(agent_1_predicate.polytope)
formula   : Formula =  (FOp(20,25) >> formation_predicate) & (GOp(120.,150.) >> agent_1_predicate) & (GOp(100.,150.) >> agent_2_predicate) & (GOp(112.,150.) >> agent_3_predicate)
# formula   : Formula =  (GOp(120.,150.) >> agent_1_predicate) & (GOp(100.,150.) >> agent_2_predicate) & (GOp(112.,150.) >> agent_3_predicate)
state_dims  = formula.get_state_dimension_of_each_predicate()
print("State dimensions of each predicate in the formula:")
print(state_dims)



time_varying_constraints1,robustness_1,kappa_gain   = compute_polyhedral_constraints(formula            = formula,
                                                                          workspace          = mas.workspace, 
                                                                          system             = mas,
                                                                          input_bounds       = mas.inputbounds,
                                                                          x_0                = x_0,
                                                                          solver             = "MOSEK",
                                                                          plot_results       = True,
                                                                          relax_input_bounds = False)

controller = AnalyticalQp( system      = mas,
                           task_constraints = time_varying_constraints1,
                           kappa       = kappa_gain,
                           add_collision_gradient = True)


# Initiate loop of the controller 
t = 0.0
Ad, Bd = mas.c2d()
state_trajectory = x_0[:,np.newaxis]
while t < 150.0:
    try :
        u = controller.get_input(x_0, t)
    except Exception as e:
        print("Controller failed at time {:.2f} with error {}".format(t,e))
        break
    print(f"At time {t:.2f}, the control action is {u}")
    x_0 = Ad @ x_0 + Bd @ u
    state_trajectory = np.hstack((state_trajectory, x_0[:,np.newaxis]))
    t   += mas.dt

# plot states of the agents over time 
time_vector = np.arange(0.0, t+mas.dt, mas.dt)
fig, ax = plt.subplots(figsize = (6,9))
for i in range(num_agents):
    ax.scatter(state_trajectory[i*2,:], state_trajectory[(i)*2 + 1,:], c=cm.plasma(time_vector/150), label = f"agent {i+1} x")
    ax.scatter(state_trajectory[i*2,0], state_trajectory[(i)*2 + 1,0], marker = "o", color = "black", s=100)  # start
    ax.scatter(state_trajectory[i*2,-1], state_trajectory[(i)*2 + 1,-1], marker = "o", color = "red", s=100)  # start
    ax.set_xlabel("x Position [m]")
    ax.set_ylabel("y Position [m]")
    ax.legend()
    ax.grid()

fig.suptitle("Position of each agent over time")
plt.tight_layout()
plt.show()