from stl_tool.stl import Formula, GOp, FOp
from stl_tool.stl.parameter_optimizer import TasksOptimizer
from stl_tool.stl.predicate_models import BoxBound
from stl_tool.polytope import Box2d,Box3d


from matplotlib import pyplot as plt
from openmpc import LinearSystem
import numpy as np

box_predicate = BoxBound(n_dim=3, size = 4, center = np.array([0., 0.,0.]))
workspace     = Box3d(x = 0,y = 0, z = 0,size = 10) 
input_bounds  = Box3d(x = 0,y = 0, z = 0,size = 10) 

print(workspace.sample_random())

A = np.zeros((3, 3))
B = np.eye(3)

system = LinearSystem.c2d(A, B, dt=0.1)

# create 3d axis
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10., 10)
ax.set_ylim(-10., 10)
ax.set_zlim(-10., 10)

bax    = box_predicate.plot(ax = ax,alpha = 0.3)

formula = ((GOp(2,4) >> box_predicate))

# formula.show_graph()


x_0 = workspace.sample_random()
scheduler = TasksOptimizer(formula, workspace)
scheduler.make_time_schedule()
# scheduler.plot_time_schedule()

solver_stats = scheduler.optimize_barriers( input_bounds = input_bounds,
                                        system       = system,
                                        x_0          = x_0)
  



scheduler.save_polytopes(filename= "test_polytopes")

# pause for e second




# plt.title("Box Predicate")
plt.show()



