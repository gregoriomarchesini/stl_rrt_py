from stl_tool.stl import Formula, GOp, FOp
from stl_tool.stl.parameter_optimizer import TasksOptimizer
from stl_tool.stl.predicate_models import BoxPredicate
from stl_tool.polytope import Box2d




from matplotlib import pyplot as plt
import numpy as np
from openmpc import LinearSystem

box_predicate = BoxPredicate(n_dim=2, size = 4, center = np.array([0, 0]))
workspace     = Box2d(x = 0,y = 0,w= 10, h = 10)
input_bounds  = Box2d(x = 0,y = 0,w= 10, h = 10) 

print(workspace.sample_random())

A = np.zeros((2, 2))
B = np.eye(2)

system = LinearSystem.c2d(A, B, dt=0.1)

fig,ax = plt.subplots(figsize=(8, 8))
bax    = box_predicate.plot(alpha = 0.3)

formula = ((GOp(2,4) >> box_predicate))

# formula.show_graph()


x_0 = workspace.sample_random()
scheduler = TasksOptimizer(formula, workspace)
scheduler.make_time_schedule()
# scheduler.plot_time_schedule()

status = scheduler.optimize_barriers( input_bounds = input_bounds,
                                        system       = system,
                                        x_0          = x_0)
  





# pause for e second




# plt.title("Box Predicate")
plt.show()



