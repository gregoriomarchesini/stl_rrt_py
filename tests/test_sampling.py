import numpy as np
from matplotlib import pyplot as plt
from stl_tool.polytope import Box2d


fig,ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# print("Vertices of the polytope:")
# print(vertices)
# polytope.plot()

box = Box2d(0, 0, 1, 1)
box.plot(ax=ax, color='red', alpha=0.5)
samples = box.sample_random(2000)
# scatter
ax.scatter(samples[0], samples[1], color='blue', alpha=0.5)


plt.show()