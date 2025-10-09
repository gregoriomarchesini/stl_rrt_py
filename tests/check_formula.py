from   matplotlib import pyplot as plt
import numpy as np
from stl_tool.stl.linear_system import ContinuousLinearSystem, MultiAgentSystem, SingleIntegrator3d

from stl_tool.stl                     import (GOp, 
                                              FOp, 
                                              ContinuousLinearSystem, 
                                              BoxPredicate2d,
                                              compute_polyhedral_constraints,
                                              TimeVaryingConstraint,
                                              Formula,
                                              is_dnf)


import os 



##########################################################
# system and dynamics
##########################################################

A             = (np.random.rand(2,2)-1)*0.1
B             = np.diag([1.,1.])
dt            = 1.0
system        = ContinuousLinearSystem(A, B, dt = dt)
max_input     = 5.
print("System max input bounds:", max_input)
print("System dynamics:", A)

system.add_state_naming("x", 0)
system.add_state_naming("y", 1)
system.add_state_naming("pos", [0,1])

# first interest point

p1 = system.get_box_predicate_on_state_name(size           = 3, 
                                            center         = np.array([1,2]), 
                                            state_name     = "pos", 
                                            predicate_name = "interest_1")
p2 = system.get_box_predicate_on_state_name(size           = 3, 
                                            center         = np.array([15,1]), 
                                            state_name     = "pos", 
                                            predicate_name = "interest_2")

p3 = system.get_box_predicate_on_state_name(size           = 3, 
                                            center         = np.array([10,18]), 
                                            state_name     = "pos", 
                                            predicate_name = "interest_3")

c_station = system.get_box_predicate_on_state_name(size           = 4, 
                                                   center         = np.array([10,10]), 
                                                   state_name     = "pos", 
                                                   predicate_name = "charging_station")


formula1      :Formula =  (FOp(20,25) >> p1)  & ((FOp(150,155) >> p2) & (GOp(0.01,200) >>  (FOp(0,140) >> c_station)) & (GOp(260,265) >> p3))
formula2      :Formula =  (FOp(20,25) >> p2)  & (FOp(150,155) >> p3) & (GOp(0.01,200) >>  (FOp(0,140) >> c_station)) & (GOp(260,265) >> p1)

formula = formula1 | formula2

print("Is formula in DNF?", is_dnf(formula))
print("Agents involved in the formula:", formula.systems_in_the_formula())
formula.show_graph()
plt.show()