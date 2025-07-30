from   matplotlib import pyplot as plt
import numpy as np
import os
np.random.seed(3)

from stl_tool.stl                     import MultiAgentSystem,compute_polyhedral_constraints
from stl_tool.environment             import Map
from stl_tool.polyhedron              import Box2d, BoxNd

import os 



# select the number of agents
num_agents = 6
agent_dim  = 2 # don't touch this

##########################################################
# Create work space and mapo
##########################################################
size = 20
workspace     = BoxNd(n_dim = agent_dim, size = size)**num_agents

##########################################################
# system and dynamics
##########################################################

max_input          = 5.
multi_agent_system = MultiAgentSystem(num_agents = num_agents,
                                      agent_dim  =  agent_dim,
                                      input_bound = max_input,)

##########################################################
# STL specifications
##########################################################

multi_agent_system.add_edge_task(agent1    = 0, 
                                 agent2    = 1,
                                 task_type ="G",
                                 start     = 10,
                                 end       = 20,
                                 center    = np.zeros(agent_dim),
                                 size      = 2.0,)

multi_agent_system.add_edge_task(agent1    = 2, 
                                 agent2    = 3,
                                 task_type ="G",
                                 start     = 10,
                                 end       = 20,
                                 center    = np.zeros(agent_dim),
                                 size      = 2.0,)

multi_agent_system.add_edge_task(agent1    = 1, 
                                 agent2    = 3,
                                 task_type ="G",
                                 start     = 10,
                                 end       = 20,
                                 center    = np.zeros(agent_dim),
                                 size      = 2.0,)







# # ##########################################################
# # # From STL to Barriers
# # ##########################################################
x_0       = np.random.uniform(low=-size/2, high=size/2, size=(num_agents, agent_dim)).flatten()


time_varying_constraints1,robustness_1   = compute_polyhedral_constraints(formula      = multi_agent_system.get_global_formula(),
                                                                          workspace    = workspace, 
                                                                          system       = multi_agent_system.get_system(),
                                                                          input_bounds = multi_agent_system.input_bound(),
                                                                          x_0          = x_0,
                                                                          plot_results = True)

