from stl_tool.polytope import Polytope, Box2d
from matplotlib import pyplot as plt
import numpy as np


pp = Box2d(x = 3,y = 0,size = 15)

A = pp.A
b = pp.b

fig,ax = plt.subplots()
pp.plot(ax)

b1 = b + np.ones(b.shape)*1.2 
pp1 = Polytope(A, b1)
pp1.plot(ax,color='red', alpha=0.5)

ax.set_aspect('equal')



pp = Box2d(x = 0,y = 0,size = 15)

A = pp.A
b = pp.b

fig,ax = plt.subplots()
pp.plot(ax)

b1 = b + np.ones(b.shape)*1.2
pp1 = Polytope(A, b1)
pp1.plot(ax,color='red', alpha=0.5)

ax.set_aspect('equal')

plt.show()
