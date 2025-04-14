import cvxpy as cp 
import numpy as np
 
k = cp.Parameter(pos= True)
a = cp.Variable(2, name='a')

A = np.random.rand(2,2)
I = np.eye(2)

mat = (A-I*k)@(A-I*k)
print(mat.is_dpp())

mat2 = cp.power((A-I*k),2)
print((k**2).is_dpp())

A_cvx = cp.Constant(A)
I_cvx = cp.Constant(np.eye(2))

mat = (A_cvx - I_cvx * k) @ (A_cvx - I_cvx * k)
print(mat.is_dpp())  