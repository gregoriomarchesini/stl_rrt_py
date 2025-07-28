import mosek.fusion as mf

def solve_lp_with_mosek():
    # Create a model
    M = mf.Model("simple_lp")

    # Define decision variable x ∈ ℝ³, constrained to be nonnegative
    x = M.variable("x", 3, mf.Domain.greaterThan(0.0))

    # Objective function: minimize c^T x
    c = [1.0, 2.0, 3.0]
    M.objective("obj", mf.ObjectiveSense.Minimize, mf.Expr.dot(c, x))

    # Constraints: A x <= b
    A = [[1.0, 1.0, 0.0],
         [0.0, 2.0, 3.0]]
    b = [1.0, 1.0]
    M.constraint("ineq", mf.Expr.mul(A, x), mf.Domain.lessThan(b))

    # Solve the problem
    M.solve()

    # Get and print the solution
    solution = x.level()
    print("Optimal solution x =", solution)

if __name__ == "__main__":
    solve_lp_with_mosek()
