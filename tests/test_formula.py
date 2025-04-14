from stl_tool.stl import Formula, GOp, FOp
from stl_tool.stl.parameter_optimizer import TasksOptimizer
from stl_tool.stl.predicate_models import BoxBound


from matplotlib import pyplot as plt
import numpy as np

box_predicate = BoxBound(n_dim=2, size = 3, center = np.array([0, 0]))


# fig,ax = plt.subplots(figsize=(8, 8))

bax = box_predicate.plot(alpha = 0.3)

formula = ((GOp(2,4) >> box_predicate) & (GOp(10,15) >> box_predicate)) & (GOp(18,20) >> (box_predicate & box_predicate))  & (FOp(25,30) >> box_predicate)

formula.show_graph()

scheduler = TasksOptimizer(formula)
scheduler.make_time_schedule()
scheduler.plot_time_schedule()




plt.title("Box Predicate")
plt.show()



