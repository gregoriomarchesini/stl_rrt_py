import cvxpy as cp

a = cp.Variable(pos=True)
b = cp.Variable(pos=True)

b.value = 3.
cost =- cp.log(a)

problem = cp.Problem(cp.Minimize(cost))
problem.solve()

print("Optimal value:", problem.value)
print("Optimal a:", a.value)
print("Optimal b:", b.value)
