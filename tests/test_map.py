from stl_tool.stl import Formula, GOp, FOp
from stl_tool.stl.parameter_optimizer import TasksOptimizer
from stl_tool.stl.predicate_models import BoxPredicate
from stl_tool.env.map import Map2d
from stl_tool.polytope import Box2d, Box3d, Polytope
from openmpc import LinearSystem

from matplotlib import pyplot as plt
import numpy as np

box_predicate  = BoxPredicate(n_dim=2, size = 3, center = np.array([0, 0]))
box_predicate1 = BoxPredicate(n_dim=2, size = 3, center = np.array([2, 0]))

box_predicate1.plot(alpha = 0.3)

formula = ((GOp(2,4) >> box_predicate1) & (GOp(10,15) >> box_predicate1)) & (GOp(18,20) >> (box_predicate & box_predicate))  & (FOp(25,30) >> box_predicate)

work_space = Box2d(0, 0, 10, 10)


scheduler = TasksOptimizer(formula, workspace=work_space)
scheduler.make_time_schedule()
scheduler.plot_time_schedule()




# obstacles
obstacles = [
    Box2d(3, 3, 5, 1),
    Box2d(5, 5, 2, 2),
    Box2d(7, 7, 1, 1),
]

# create map
map = Map2d(x_lim=(-10, 10), y_lim=(-10, 10))
map.add_obstacle(obstacles)
# draw
map.draw_formula_predicate(formula)
formula.show_graph()
plt.show()



