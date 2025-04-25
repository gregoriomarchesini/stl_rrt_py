from   matplotlib import pyplot as plt
import numpy as np
np.random.seed(3)

from stl_tool.stl                     import GOp, FOp, TasksOptimizer, ContinuousLinearSystem, BoxBound2d, BoxBound
from stl_tool.environment             import Map
from stl_tool.polytope                import Box2d

from stl_tool.planners import StlRRTStar
from json import loads
import os 




workspace     = Box2d(x = 0,y = 0,size = 20) # cartesian product of a square 10x10 
map           = Map(workspace = workspace)

# load obstacles 
file_path = os.path.join(os.path.dirname(__file__), "map2d.json")
map_json = loads(open(file_path,mode="r").read())

for object in map_json:
    if object["name"].split("_")[0] == "obstacle":
        center = np.array([object["center_x"], object["center_y"]])
        size   = np.array([object["size_x"], object["size_y"]])
        obstacle = Box2d(x = center[0],y = center[1],size = size)
        obstacle_multi_agent = obstacle

        map.add_obstacle(obstacle_multi_agent)

map.draw() # draw if you want :)
map.enlarge_obstacle(border_size=0.2) # enlarge obstacles

named_map = {item["name"]: item for item in map_json}



# first interest point
intrest_point = named_map["goal1"]
goal1 = BoxBound2d(size = [intrest_point["size_x"],intrest_point["size_y"]] , center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "goal1")
# second interest point
intrest_point = named_map["goal2"]
goal2 = BoxBound2d(size = [intrest_point["size_x"],intrest_point["size_y"]] , center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "goal2")
# third interest point
intrest_point = named_map["goal3"]
goal3 = BoxBound2d(size = [intrest_point["size_x"],intrest_point["size_y"]] , center = np.array([intrest_point["center_x"], intrest_point["center_y"]]), name = "goal3")


# gathering_point = BoxBound(dims =[0,1,2,3,4,5], size = 1.3, center= np.array([0., 0., 0., 0., 0., 0.]), name = "gathering_point")

formula       =  (FOp(130,140) >> goal1)  & (FOp(130,140) >> goal2) & (FOp(130,140) >> goal3) 


fig,ax = map.draw_formula_predicate(formula = formula, alpha =0.2)
plt.show()