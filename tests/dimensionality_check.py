from stl_tool.stl import Formula, GOp, FOp
from stl_tool.stl.parameter_optimizer import TasksOptimizer
from stl_tool.stl.predicate_models import BoxPredicate
from stl_tool.polytope import Box2d,Box3d, BoxNd


from matplotlib import pyplot as plt
import numpy as np
from openmpc import LinearSystem


solver_time = []
for dimension in range(2,10) :

    box_predicate = BoxPredicate(n_dim=dimension, size = 4)
    workspace     = BoxNd(n_dim = dimension, size = 10)
    input_bounds  = BoxNd(n_dim = dimension, size = 10)

    print(workspace.sample_random())

    A = np.zeros((dimension, dimension))
    B = np.eye(dimension)

    system = LinearSystem.c2d(A, B, dt=0.1)

    # # create 3d axis
    # fig = plt.figure()
    # ax  = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(-10., 10)
    # ax.set_ylim(-10., 10)
    # ax.set_zlim(-10., 10)

    # bax    = box_predicate.plot(ax = ax,alpha = 0.3)

    formula = ((GOp(2,4) >> box_predicate)) & ((GOp(10,20) >> box_predicate)) & ((GOp(25,30) >> box_predicate))

    # formula.show_graph()


    x_0 = workspace.sample_random()
    scheduler = TasksOptimizer(formula, workspace)
    scheduler.make_time_schedule()
    # scheduler.plot_time_schedule()

    solver_stats = scheduler.optimize_barriers( input_bounds = input_bounds,
                                                system       = system,
                                                x_0          = x_0)
    
    solver_time += [solver_stats.solve_time]
    





fig,ax = plt.subplots()
ax.plot(range(2,10),solver_time)

plt.show()



