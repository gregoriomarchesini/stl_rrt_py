import casadi as ca


opti = ca.Opti("conic")
x    = opti.variable(2)
y = opti.variable(2)
opti.set_initial(x, [1, 2])

p = opti.parameter(2)
opti.set_value(p, [3, 4])

opti.solver("osqp")
opti.minimize(ca.sumsqr(x + y - p))
opti.solve()
print(opti.stats())
print(opti)