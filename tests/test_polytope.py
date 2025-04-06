from stl_tool.polytope.polytope import Polytope
import numpy as np
from matplotlib import pyplot as plt


A = np.array([[1, 0,1.], [0, 1, 2.], [-1, 0,3.], [4,0, -1]])
b = np.array([1, 1, 1, 1])

polytope = Polytope(A, b)
fig,ax   = plt.subplots(figsize=(8, 8))

polytope.plot(ax = ax,alpha=0.5, color='blue')


plt.show()