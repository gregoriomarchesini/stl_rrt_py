from stl_tool.stl.linear_system import MultiAgentSystem, ContinuousLinearSystem, RelativeFormationTuple
import numpy as np

A = RelativeFormationTuple(system_1="agent1", system_2="agent2", state_name_1="pos", state_name_2="pos", center=np.array([1.0, 0.0]), size=2.0)
A = RelativeFormationTuple(system_1="agent1", system_2="agent2", dims_1=2, state_name_2="pos", center=np.array([1.0, 0.0]), size=2.0)
A = RelativeFormationTuple(system_1="agent1", system_2="agent2", dims_1=2, dims_2 = [2,3], center=np.array([1.0, 0.0]), size=2.0)
print(A)


