import random
import numpy as np
import matplotlib.pyplot as plt
from   typing          import TypedDict
from   tqdm            import tqdm
from   scipy.spatial   import KDTree
from scipy.interpolate import BSpline

def generate_bspline(p_start, p_end, t_start, t_end, num_points=20):
    """
    Generate a B-spline with 4 control points between start and end points in space-time
    p_start: starting point in space (x,y,z)
    p_end: ending point in space (x,y,z)
    t_start: starting time
    t_end: ending time
    """
    # Combine space and time dimensions
    start = np.append(p_start, t_start)
    end = np.append(p_end, t_end)
    
    # Create 4 control points (including start and end)
    # You might want to add some intermediate points for smoother curves
    ctrl_pts = np.array([
        start,
        start + 0.33*(end - start),
        start + 0.67*(end - start),
        end
    ])
    
    # Create knot vector for cubic B-spline (degree=3)
    knots = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Create B-spline object
    degree = 3
    spline = BSpline(knots, ctrl_pts, degree)
    
    # Evaluate at parameter values
    t = np.linspace(0, 1, num_points)
    trajectory = spline(t)
    
    return trajectory[:, :3], trajectory[:, 3]  # space coords, time coords



start= np.array([0, 0, 0])
end = np.array([10, 10, 10])
t_start = 0
t_end = 10
num_points = 20
trajectory, time_coords = generate_bspline(start, end, t_start, t_end, num_points)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='B-spline trajectory')
ax.scatter(start[0], start[1], start[2], color='r', label='Start')
ax.scatter(end[0], end[1], end[2], color='g', label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()