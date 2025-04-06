import cvxpy as cp

a = cp.Variable(pos=True)
b = cp.Variable(pos=True)

a.value = 10
b.value = 10

cost =  cp.exp( - (a - b) )


constraints = [a ==4, b == 3]
problem = cp.Problem(cp.Minimize(cost),constraints=constraints)
problem.solve(verbose = True)

print("Optimal value:", problem.value)
print("Optimal a:", a.value)
print("Optimal b:", b.value)

print(type(a.value))
