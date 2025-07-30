import cvxpy as cp
import numpy as np
from scipy import sparse
from time import perf_counter

# Problem size
n = 1000  # number of variables
m = 100000  # number of constraints
density = 0.01  # sparsity level

# Random problem data
np.random.seed(0)
c = np.random.randn(n)
b = np.random.randn(m)

# DENSE matrix
A_dense = np.random.randn(m, n) * (np.random.rand(m, n) < density)

# SPARSE matrix
A_sparse = sparse.csc_matrix(A_dense)

def solve_lp(A, label):
    x = cp.Variable(n)
    constraints = [A @ x <= b]
    objective = cp.Minimize(c @ x)
    prob = cp.Problem(objective, constraints)

    start = perf_counter()
    prob.solve(solver=cp.MOSEK, verbose=True)
    end = perf_counter()

    print(f"{label:<10} | Status: {prob.status:<10} | Canonicalization + Solve Time: {end - start:.4f} s")

# Run both versions
print("Dense Problem")
solve_lp(A_dense, "Dense")
print("Sparse Problem")
solve_lp(A_sparse, "Sparse")
