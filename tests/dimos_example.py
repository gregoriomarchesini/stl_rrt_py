from   matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import numpy as np
import os 
from matplotlib.animation import FuncAnimation
from stl_tool.polyhedron import Polyhedron

from stl_tool.stl                     import (GOp, 
                                              FOp, 
                                              SingleIntegrator2d,
                                              MultiAgentSystem,
                                              RelativeFormationTuple,
                                              compute_polyhedral_constraints)

from stl_tool.stl.logic     import Formula
from stl_tool.controllers   import MPCProblem, TimedMPC
"""
Design of time-varying sets for mas of single integrators
"""
##########################################################
# Create work space 
##########################################################
input_bounds = 5.5
num_agents   = 1
systems      = []
dt           = 0.2
for agents in range(num_agents):
    system = SingleIntegrator2d(name = f"agent_{agents+1}", dt = dt)
    system.set_workspace_bounds(ubx = [10., 10.],lbx = [-10., -10.])
    system.set_input_bounds(ubu = [input_bounds, input_bounds],lbu = [-input_bounds, -input_bounds])
    systems.append(system)
    system.print_states_names()

mas = MultiAgentSystem(systems = systems)
print("The workspace of the multi-agent system is:")
print(mas.workspace)
print("Input bounds of each agent are:")
print(mas.inputbounds)


##########################################################
# STL specifications
##########################################################
size = 2.0
c1 = np.array([-6.0, 5.0])
predicate_1   = mas.get_single_agent_box_predicate_by_name( system_name = "agent_1",
                                                                  center      = c1,
                                                                  size        = [size, size],
                                                                  state_name  = "position")
c2 = np.array([4.0, 6.0])
predicate_2   = mas.get_single_agent_box_predicate_by_name( system_name = "agent_1",
                                                                  center      = c2,
                                                                  size        = [size, size],
                                                                  state_name  = "position")

c3 = np.array([0.0, 0.0])
predicate_3   = mas.get_single_agent_box_predicate_by_name( system_name = "agent_1",
                                                                  center      = c3,
                                                                  size        = [size, size],
                                                                  state_name  = "position")

print("Predicate 1 is: ", predicate_1)


x_0 = np.array([ 0.0,  0.0 ])

formula   : Formula =  (GOp(20,25) >> predicate_1) & (FOp(50.,55.) >> predicate_2) & (GOp(0.1,70.)>> (FOp(0.,40.) >> predicate_3))

state_dims  = formula.get_state_dimension_of_each_predicate()
print("State dimensions of each predicate in the formula:")
print(state_dims)



time_varying_constraints1,robustness_1   = compute_polyhedral_constraints(formula            = formula,
                                                                          workspace          = mas.workspace, 
                                                                          system             = mas,
                                                                          input_bounds       = mas.inputbounds,
                                                                          x_0                = x_0,
                                                                          solver             = "MOSEK",
                                                                          plot_results       = True,
                                                                          relax_input_bounds = False)


mpc_parameters = MPCProblem( system   = mas,
                             horizon  = 10,
                             Q        = np.zeros((mas.state_dim,mas.state_dim)),
                             R        = 10*np.eye(mas.input_dim),
                             QT       = np.zeros((mas.state_dim,mas.state_dim)),
                             solver   = "MOSEK")

# add constraints
for tvc in time_varying_constraints1 :
    mpc_parameters.add_general_state_time_constraints(Hx = tvc.H, bx = tvc.b, start_time = tvc.start_time, end_time = tvc.end_time, is_hard=True)

mpc_parameters.add_general_input_constraints(Hu = mas.inputbounds.A, bu = mas.inputbounds.b, is_hard=True)


mpc_controller = TimedMPC( mpc_params = mpc_parameters)

# Initiate loop of the controller
t = 0.0
Ad, Bd = mas.c2d()
state_trajectory = x_0[:,np.newaxis]
t_vec=t 
while t < 125.0:
    try :
        u = mpc_controller.get_control_action(x0 = x_0, t0 = t, reference = np.zeros(mas.state_dim))
    except Exception as e:
        print("MPC failed at time {:.2f} with error {}".format(t,e))
        break
    print(f"At time {t:.2f}, the control action is {u}")
    x_0 = Ad @ x_0 + Bd @ u
    state_trajectory = np.hstack((state_trajectory, x_0[:,np.newaxis]))
    t_vec = np.hstack((t_vec, t+dt))
    t   += mas.dt
    print("current time:", t)

# plot states of the agents over time 
time_vector = np.arange(0.0, t+mas.dt, mas.dt)
fig, ax = plt.subplots(figsize = (6,9))
for i in range(num_agents):
    ax.scatter(state_trajectory[i*2,:], state_trajectory[(i)*2 + 1,:], c=cm.plasma(t_vec/t_vec[-1]), label = f"agent {i+1} x")
    ax.scatter(state_trajectory[i*2,0], state_trajectory[(i)*2 + 1,0], marker = "o", color = "black", s=100)  # start
    ax.scatter(state_trajectory[i*2,-1], state_trajectory[(i)*2 + 1,-1], marker = "o", color = "red", s=100)  # start
    ax.set_xlabel("x Position [m]")
    ax.set_ylabel("y Position [m]")
    ax.legend()
    ax.grid()




fig, ax = plt.subplots()

# add rectangle patch for each predicate
rext1 = Rectangle((c1[0]-size/2, c1[1]-size/2), size, size, linewidth=1, edgecolor='r', facecolor='r', label = "predicate 1", alpha=0.5)
rext2 = Rectangle((c2[0]-size/2, c2[1]-size/2), size, size, linewidth=1, edgecolor='g', facecolor='g', label = "predicate 2", alpha=0.5)
rext3 = Rectangle((c3[0]-size/2, c3[1]-size/2), size, size, linewidth=1, edgecolor='b', facecolor='b', label = "predicate 3", alpha=0.5)
ax.add_patch(rext1)
ax.add_patch(rext2)
ax.add_patch(rext3)

poly_patches = dict()
# initialize one Rectangle patch per constraint
for constraint in time_varying_constraints1:
    # start with a zero-size rectangle
    patch = Rectangle(
        (0, 0),              # bottom-left corner
        0, 0,                # width, height
        angle=0.0,           # rotation angle
        color='skyblue',
        alpha=0.5
    )
    ax.add_patch(patch)
    poly_patches[constraint] = patch

colors = cm.plasma(np.linspace(0, 1, num_agents))
agents_scatter = []
for i in range(num_agents):
    scat = ax.scatter([], [], color=colors[i], s=60, label=f"Agent {i+1}")
    ax.scatter(state_trajectory[2*i, 0], state_trajectory[2*i+1, 0],
               marker='o', color='black', s=100)  # start
    ax.scatter(state_trajectory[2*i, -1], state_trajectory[2*i+1, -1],
               marker='o', color='red', s=100)    # goal
    agents_scatter.append(scat)

ax.legend()

ax.set_xlim([-9,9])
ax.set_ylim([-9,9])


def rectangle_geometry_from_vertices(vertices):
    """
    Given 4 rectangle vertices (Nx2), return center, width, height, and rotation angle [deg].
    Returns (center, width, height, angle)
    """
    if vertices.shape[0] < 4:
        # degenerate or inactive constraint
        return np.zeros(2), 0.0, 0.0, 0.0

    # ensure consistent order (counterclockwise)
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])
    vertices = vertices[np.argsort(angles)]

    # side vectors
    v1 = vertices[1] - vertices[0]
    v2 = vertices[3] - vertices[0]

    width = np.linalg.norm(v1)
    height = np.linalg.norm(v2)
    angle = np.degrees(np.arctan2(v1[1], v1[0]))

    return center, width, height, angle


# --- Main loop ---
rect_params_per_constraint = dict()

for constraint in time_varying_constraints1:
    timed_dict = {}
    for t in t_vec:
        if constraint.start_time <= t <= constraint.end_time:
            h = constraint.H[:, -1]
            b = constraint.b
            bb = b - h * t
            A = constraint.H[:, :-1]
            poly = Polyhedron(A, bb)
            vertices = poly.vertices

            center, width, height, angle = rectangle_geometry_from_vertices(vertices)
            timed_dict[t] = dict(center=center, width=width, height=height, angle=angle)
        else:
            # inactive time: placeholder zero rectangle
            timed_dict[t] = dict(center=np.zeros(2), width=0.0, height=0.0, angle=0.0)

    rect_params_per_constraint[constraint] = timed_dict
    

# --- Update function for animation ---
def update(frame):
    t = t_vec[frame]

    # Update rectangles
    for constraint in time_varying_constraints1:
        params = rect_params_per_constraint[constraint][t]
        cx, cy = params["center"]
        w, h = params["width"], params["height"]
        angle = params["angle"]

        x0 = cx - w / 2
        y0 = cy - h / 2

        patch  = poly_patches[constraint]
        patch.set_xy([x0, y0])  # bottom-left as list or tuple
        patch.set_width(w)
        patch.set_height(h)
        patch.angle = angle

    # Update agents
    for i in range(num_agents):
        xy = np.column_stack((state_trajectory[2*i, :frame],
                      state_trajectory[2*i+1, :frame]))
        agents_scatter[i].set_offsets(xy)

    return list(poly_patches.values()) + agents_scatter

frame_indices = np.arange(1, len(t_vec), 5)  # e.g., every 5th frame
ani = FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=True)
ani.save("trajectory_animation.mp4", writer="ffmpeg", fps=10, dpi=200)
plt.show()
