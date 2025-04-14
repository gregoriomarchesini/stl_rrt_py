from   matplotlib import pyplot as plt
import numpy as np

from stl_tool.stl                     import GOp, FOp, TasksOptimizer, BoxBound, ContinuousLinearSystem
from stl_tool.environment.map         import Map
from stl_tool.polytope                import Box2d, BoxNd, Box3d




workspace = BoxNd(n_dim= 4, size = [15,15,4,4])
map4d       = Map(workspace = workspace)

# obstacles
obstacles = [
    Box3d(x = 3,y=3, z= 3, size = [5, 1, 3]),
    Box3d(x = 5,y=5, z= 3, size = [2, 2, 3]),
    Box3d(x = 7,y=7, z= 3, size = [1, 1, 3]),
]

map4d.add_obstacle(obstacles)
map4d.draw(projection_dim=[0,1,4])

# # create map
# map = Map2d(x_lim=(-10, 10), y_lim=(-10, 10))
# map.add_obstacles(obstacles)
# # draw
# map.draw_formula_predicate(formula)
# formula.show_graph()
plt.show()



