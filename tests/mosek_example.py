
import mosek.fusion as mf
import mosek.fusion.pythonic
import numpy as np
import scipy.sparse as sp

# Create a new model
with mf.Model("simple_lp") as M:
    # Define 2 variables (x1, x2) with lower bound 0
    m = 10000
    x = M.variable([m,10], mf.Domain.greaterThan(0.0))
    b = np.ones((m,10))
    
    A                = sp.random(m, m, density= 0.2, format='coo', dtype=np.float64)
    rows, cols, vals = A.row, A.col, A.data
    Af               = mf.Matrix.sparse( m, m, rows, cols, vals)



    

    M.constraint(Af@x >= b)
