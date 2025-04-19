import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stlpy.systems import SingleIntegrator
from stlpy import Predicate

# Your JSON data pasted here:
json_data = [ ... ]  # <== paste your JSON list here

# Initialize system
sys = SingleIntegrator(2)
predicates = {}

# Plot environment and define predicates
fig, ax = plt.subplots(figsize=(10, 10))

for obj in json_data:
    cx, cy = obj["center_x"], obj["center_y"]
    sx, sy = obj["size_x"], obj["size_y"]
    x_min, x_max = cx - sx / 2, cx + sx / 2
    y_min, y_max = cy - sy / 2, cy + sy / 2

    # Plotting
    color = 'red' if obj["type"] == "obstacle" else 'green'
    ax.add_patch(Rectangle((x_min, y_min), sx, sy, edgecolor='black', facecolor=color, alpha=0.5))
    ax.text(cx, cy, obj["name"], ha='center', va='center', fontsize=8)

    # Predicate construction
    A = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])
    b = np.array([x_max, -x_min, y_max, -y_min])

    pred = Predicate(obj["name"], A, b)
    predicates[obj["name"]] = pred

ax.set_title("Environment with Predicates")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.axis('equal')
ax.grid(True)
plt.show()
