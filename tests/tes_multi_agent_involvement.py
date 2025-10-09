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

from stl_tool.stl.logic     import Formula, is_dnf
from stl_tool.controllers   import MPCProblem, TimedMPC
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
    system = SingleIntegrator2d(name = f"agent_{agents+1}", dt = 0.05)
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



formation_predicate = mas.get_relative_formation_box_predicate(formation_1, formation_2)
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

agent_4_predicate   = mas.get_single_agent_box_predicate_by_name( system_name = "agent_4",
                                                                  center      = np.array([0.0, 0.0]),
                                                                  size        = [1.0, 1.0],
                                                                  state_name  = "position")



x_0 = np.array([ -10.0,  8.0,   # agent 1 position
                  8.0,  6.0,   # agent 2 position
                7.0,  4.0,
                6.0,  2.0])  # agent 3 position


formula   : Formula =  ((FOp(20,25) >> formation_predicate) & (GOp(120.,150.) >> agent_1_predicate)) |  ((FOp(20,25) >> formation_predicate) & (GOp(120.,150.) >> agent_4_predicate)) 

state_dims  = formula.get_state_dimension_of_each_predicate()
print("State dimensions of each predicate in the formula:")
print(state_dims)
# check if formula is in DNF
print("Is the formula in DNF?", is_dnf(formula))
# check agents involved in the formula
print("Agents involved in the formula:", formula.systems_in_the_formula())
formula_1 = Formula(formula.root.children[0])
formula_2 = Formula(formula.root.children[1])

print("Agents involved in the first disjunct:", formula_1.systems_in_the_formula())
print("Agents involved in the second disjunct:", formula_2.systems_in_the_formula())



# time_varying_constraints1,robustness_1   = compute_polyhedral_constraints(formula            = formula,
#                                                                           workspace          = mas.workspace, 
#                                                                           system             = mas,
#                                                                           input_bounds       = mas.inputbounds,
#                                                                           x_0                = x_0,
#                                                                           solver             = "MOSEK",
#                                                                           plot_results       = True,
#                                                                           relax_input_bounds = False)


# mpc_parameters = MPCProblem( system   = mas,
#                              horizon  = 10,
#                              Q        = np.zeros((mas.state_dim,mas.state_dim)),
#                              R        = 0.1*np.eye(mas.input_dim),
#                              QT       = np.zeros((mas.state_dim,mas.state_dim)),
#                              solver   = "CLARABEL")

# # add constraints
# for tvc in time_varying_constraints1 :
#     mpc_parameters.add_general_state_time_constraints(Hx = tvc.H, bx = tvc.b, start_time = tvc.start_time, end_time = tvc.end_time, is_hard=False)

# mpc_parameters.add_general_input_constraints(Hu = mas.inputbounds.A, bu = mas.inputbounds.b, is_hard=True)


# mpc_controller = TimedMPC( mpc_params = mpc_parameters)


# # Initiate loop of the controller 
# t = 0.0
# Ad, Bd = mas.c2d()
# state_trajectory = x_0[:,np.newaxis]
# while t < 150.0:
#     try :
#         u = mpc_controller.get_control_action(x0 = x_0, t0 = t, reference = np.zeros(mas.state_dim))
#     except Exception as e:
#         print("MPC failed at time {:.2f} with error {}".format(t,e))
#         break
#     print(f"At time {t:.2f}, the control action is {u}")
#     x_0 = Ad @ x_0 + Bd @ u
#     state_trajectory = np.hstack((state_trajectory, x_0[:,np.newaxis]))
#     t   += mas.dt

# # plot states of the agents over time 
# time_vector = np.arange(0.0, t+mas.dt, mas.dt)
# fig, ax = plt.subplots(figsize = (6,9))
# for i in range(num_agents):
#     ax.scatter(state_trajectory[i*2,:], state_trajectory[(i)*2 + 1,:], c=cm.hot(time_vector/150), label = f"agent {i+1} x")
#     ax.scatter(state_trajectory[i*2,0], state_trajectory[(i)*2 + 1,0], marker = "o", color = "black", s=100)  # start
#     ax.scatter(state_trajectory[i*2,-1], state_trajectory[(i)*2 + 1,-1], marker = "o", color = "red", s=100)  # start
#     ax.set_xlabel("x Position [m]")
#     ax.set_ylabel("y Position [m]")
#     ax.legend()
#     ax.grid()

# fig.suptitle("Position of each agent over time")
# plt.tight_layout()
# plt.show()