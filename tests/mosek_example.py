import mosek.fusion as mf
import mosek.fusion.pythonic
import numpy as np

# Create a new model
with mf.Model("simple_lp") as M:
    # Define 2 variables (x1, x2) with lower bound 0
    m = 10000
    x = M.variable([m,10], mf.Domain.greaterThan(0.0))
    b = np.ones((m,10))
    A = np.random.rand(m,m).flatten().tolist()
    Af= mf.Matrix.dense(m, m, A)
    

    M.constraint(Af@x >= b)

    