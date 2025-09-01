#!/usr/bin/env python

import numpy as np
from stl_tool.stl import (
    GOp,
    FOp,
    UOp,
    ContinuousLinearSystem,
    BoxPredicate2d,
    RegularPolygonPredicate2D,
    Geq,
)
from stl_tool.environment import Map
from stl_tool.polyhedron import Box2d, Polyhedron

import matplotlib.pyplot as plt


# ------- Create work space and map -------------
workspace = Box2d(
    x=0, y=0, size=10
)  # square 10x10 (this is a 2d workspace, so the system it refers to must be 2d)
map = Map(
    workspace=workspace
)  # the map object contains the workpace, but it also contains the obstacles of your environment.

# create obstacles
# some simple boxes
# map.add_obstacle(Box2d(x=3, y=3, size=1))

# ------- System definition -------------
# Initial conditions
x0 = [-0.4, -0.15]
# State and input bounds
dx = 1.5
du = 9.0

A = np.block([[3, 1], [1, 2]])
B = np.block([[1, 0], [1, 1]])

input_bounds = Box2d(x=0.0, y=0.0, size=du * 2)


print(input_bounds)
input_bounds.plot(color="blue", alpha=0.5)
plt.title("Input bounds")

# create a continuous linear system with the given A and B matrices
system = ContinuousLinearSystem(A, B, dt=0.1)


# --------- Predicate functions -------------
# Goal positions
pA = np.array([-0.3, 0.2])
pB = np.array([0.35, 0.5])
pC = np.array([0.35, -0.5])

# Size of the square around the goal
dA = 0.45
dB = 0.45
dC = 0.45


# creates a box over the first three dimension  of the system (so on the positon).
center = np.array([-0.0, 0.0])
hX = BoxPredicate2d(size=dx, center=center, name="State bounds")
h1 = BoxPredicate2d(size=dA, center=pA, name="Goal A")

h2C = np.array([-10, 1])
h2d = -2
# h2H = Polyhedron(h2C, h2d)
h2 = Geq(dims=[0, 1], state_dim=2, bound=h2d, name="Halfplane predicate")

h3 = BoxPredicate2d(size=dB, center=pB, name="Goal B")
h4 = BoxPredicate2d(size=dC, center=pC, name="Goal C")

# --------- Formula definition -------------
taG1 = 0.0
tbG1 = 4.0
formula1 = GOp(taG1, tbG1) >> h1

taU2 = 5.0
tbU2 = 8.0
# formula2 = (GOp(0.0, taU2)) & (FOp(taU2, tbU2) >> h3)
formula2 = FOp(taU2, tbU2) >> h3

taF3 = 8.0
tbF3 = 10.0
formula3 = FOp(taF3, tbF3) >> h4

formula = formula1 & formula2 & formula3


fig, ax = map.draw_formula_predicate(formula=formula, alpha=0.2)

plt.show()
