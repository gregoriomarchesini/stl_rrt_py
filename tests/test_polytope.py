from stl_tool.stl.polytope import Polytope
import numpy as np

A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
b = np.array([1, 1, 0, 0])

polytope = Polytope(A, b)

polytope.get_V_representation()
print(polytope.get_V_representation())

a = np.random.rand(2,4)
b = np.random.rand(2,4)

print(np.vstack((a,b)))

print({2.3,2.3})