import cvxpy as cp
import numpy as np
import casadi as ca

def factorial(n: int) -> float:
    """
    Calculate the factorial of a non-negative integer n.
    
    :param n: Non-negative integer for which to calculate the factorial.
    :return: The factorial of n.
    """
    if n == 0 or n == 1:
        return 1.0
    if n < 0:
        raise ValueError("Factorial is not defined for negative integers.")
    
    return n * factorial(n - 1)

def binomial_coefficients(n: int, k: int) -> float:
    """
    Calculate the binomial coefficient C(n, k) = n! / (k! * (n - k)!).
    
    :param n: Total number of items.
    :param k: Number of items to choose.
    :return: The binomial coefficient C(n, k).
    """
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))



class BezierCurve:
    """
    Representation of a Bezier curve of ordr n.

    The bezier curve is given by the equation

    B(t) = sum(i=0 to n) (C(n, i) * (1-t)^(n-i) * t^i * P_i)

    where C(n, i) is the binomial coefficient, P_i are the control points, and t is in [0, 1].
    The curve can be representd in matrix form as:

    B(t) =  M(t) @ P

    where P is the stacked vector of control points and M(t) is the Bernstein polynomial matrix evaluated at time t.

    Properties that are good to remember are :
    1) The curve is contained in the convex hull of the control points.
    2) The curve starts at P_0 and ends at P_n.
    3) The derivative of a bezier cruve is a bezier curve of lower order, specifically of order n-1.
    """

    def __init__(self, control_points: list[np.ndarray],): 
        """
        Initialize a Bezier curve of order n with n + 1 control points
        
        :param control_points: List of control points [P_0, ..., P_{n}]. 
        """
        
        self.control_points = control_points
        self.n = len(control_points) - 1
        if self.n < 1:
            raise ValueError("A Bezier curve must have at least 2 control points (order 1).")
        if not all(isinstance(p, np.ndarray) for p in control_points):
            raise TypeError("All control points must be numpy arrays.")
        
        # flatten all points for easier manipulation
        self.control_points = [p.flatten() for p in control_points]
        self.dim = self.control_points[0].shape[0]  # Dimension of the control points
        if not all(p.shape[0] == self.dim for p in self.control_points):
            raise ValueError("All control points must have the same dimension. But dimensions are :" + str([p.shape[0] for p in self.control_points]))
        

        self.stacked_vector = np.hstack(self.control_points)  # Stacked vector of control points

    def bernstein_matrix(self, t: float) -> np.ndarray:
        """
        Compute the Bernstein polynomial matrix for a given t.
        
        :param t: Parameter in [0, 1] at which to evaluate the Bernstein polynomial matrix.
        :return: The Bernstein polynomial matrix evaluated at t.
        """
        n = self.n
        m = np.zeros(n + 1)
        for i in range(n + 1):
            m[i] = binomial_coefficients(n, i) * ((1 - t) ** (n - i)) * (t ** i)
        
        
        I = np.eye(self.dim)
        M = np.kron(m, I)  # Kronecker product to create the matrix for each dimension (self.dim x (self.dim \cdot n + 1))
        
        return M
    

    def evaluate(self, t: float) -> np.ndarray:        
        """
        Evaluate the Bezier curve at a given parameter t.
        
        :param t: Parameter in [0, 1] at which to evaluate the curve.
        :return: The point on the Bezier curve at parameter t.
        """

        if not (0 <= t <= 1):
            raise ValueError("Parameter t must be in the range [0, 1].")
        
        M = self.bernstein_matrix(t)
        return M @ self.stacked_vector
    
    def get_derivative(self) -> "BezierCurve":
        """
        Get the derivative of the Bezier curve, which is a Bezier curve of order n-1.
        
        :return: A new BezierCurve instance representing the derivative.
        """
        if self.n == 0:
            raise ValueError("Cannot compute derivative of a constant curve (order 0).")
        
        # the control points of the derivatives are obtained as the difference of the control points of the orinal curve 
        # Namely we have that the control points of the derivative are 
        # p_1- p_0, p_2 - p_1, ..., p_n - p_{n-1} 
        # we build the matrix that does this difference

        P = np.zeros((self.n, self.n+1))
        for i in range(self.n):
            P[i, i] = -1
            P[i, i + 1] = 1
        
        
        P = np.kron(P, np.eye(self.dim))
        print(P)
        new_control_points = P @ self.stacked_vector # the difference is just a matrix operation

        new_control_points = [new_control_points[i * self.dim:(i + 1) * self.dim] for i in range(self.n)]  # reshape to control points
        return BezierCurve(new_control_points)
    
    def draw(self, ax=None, num_points=100, **kwargs):
        """
        Draw the Bezier curve on a given matplotlib axis.
        
        :param ax: Matplotlib axis to draw on. If None, a new figure and axis will be created.
        :param num_points: Number of points to evaluate the curve for drawing.
        :param kwargs: Additional keyword arguments for plotting.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        
        t_values = np.linspace(0, 1, num_points)
        points = np.array([self.evaluate(t) for t in t_values])
        
        ax.plot(points[:, 0], points[:, 1], **kwargs)
        for jj,point in enumerate(self.control_points):
            ax.scatter(point[0], point[1], color='red', s=50, edgecolor='black')
            ax.text(point[0], point[1], f'P_{jj}', fontsize=12, ha='right', va='bottom')
        ax.set_title('Bezier Curve')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')



class PBezierCurve:
    """
    An instance of a Parameteric Bezier curve where control points are allowed to be cvxpy variables as well as fixed numpy arrays.
    """

    def __init__(self, order:int, dim:int , opti : ca.Opti | None = None):
        """
        Initialize a parametric Bezier curve with parameteric control points
        
        :param order: Order of the Bezier curve (number of control points is order + 1).
        :type order: int
        :param dim: Dimension of the space in which the Bezier curve is defined (e.g. for 2D, dim=2, for 3D, dim =3).
        :type dim: int
        :param opti: Optional casadi Opti instance for optimization. If None, a new Opti instance will be created.
        :type opti: ca.Opti | None
        """

        self.dim            : int                            = dim
        self.opti           : ca.Opti                        = opti
        if self.opti is None:
            self.opti       = ca.Opti("conic")

        self.control_points : list[ca.MX]                    = [self.opti.variable(dim) for _ in range(order + 1)]
        self.n              : int                            = order  # Order of the Bezier curve

    @property
    def stacked_control_points(self) -> ca.MX:
        """
        Get the stacked vector of control points.
        
        :return: A cvxpy expression representing the stacked vector of control points.
        """
        return ca.vertcat(*self.control_points)
    

    def get_control_points_as_array(self):
        """
        Returns the current value of the control points. If a control point is a numpy array then it is retuyrned as it is.
        If it is a cvxpy variable, then it is returned as a numpy array with the current value.
        """

        return [self.opti.value(p).full().flatten() for p in self.control_points]
    
    def bernstein_matrix(self, t: float) -> np.ndarray:
        """
        Compute the Bernstein polynomial matrix for a given t.
        
        :param t: Parameter in [0, 1] at which to evaluate the Bernstein polynomial matrix.
        :return: The Bernstein polynomial matrix evaluated at t.
        """
        n = self.n
        m = np.zeros(n + 1)
        for i in range(n + 1):
            m[i] = binomial_coefficients(n, i) * ((1 - t) ** (n - i)) * (t ** i)
        
        
        I = np.eye(self.dim)
        M = np.kron(m, I)  # Kronecker product to create the matrix for each dimension (self.dim x (self.dim \cdot n + 1))
        
        return M
    
    def evaluate(self, t: float) -> ca.MX:
        """
        Evaluate the Bezier curve at a given parameter t.
        :param t: Parameter in [0, 1] at which to evaluate the curve.
        """

        if not (0 <= t <= 1):
            raise ValueError("Parameter t must be in the range [0, 1].")
        
        M = self.bernstein_matrix(t)
        return ca.mtimes(M,self.stacked_control_points)
    
    

    def get_derivative(self) -> "PBezierCurve":
        """
        Get the derivative of the Bezier curve, which is a PBezierCurve of order n-1.
        
        :return: A new PBezierCurve instance representing the derivative.
        """
        if self.n == 0:
            raise ValueError("Cannot compute derivative of a constant curve (order 0).")
        
        # the control points of the derivatives are obtained as the difference of the control points of the orinal curve 
        # Namely we have that the control points of the derivative are 
        # p_1- p_0, p_2 - p_1, ..., p_n - p_{n-1} 
        # we build the matrix that does this difference

        P = np.zeros((self.n, self.n+1))
        for i in range(self.n):
            P[i, i] = -1
            P[i, i + 1] = 1
        
        
        P = np.kron(P, np.eye(self.dim))
        new_control_points = ca.mtimes(P,self.stacked_control_points) # the difference is just a matrix operation
        new_control_points = [new_control_points[i * self.dim:(i + 1) * self.dim] for i in range(self.n)]
        
        return PBezierCurve.from_control_points(control_points= new_control_points, opti= self.opti) # return fully parameteric bezier curve with the new control points
    
    @classmethod
    def from_control_points(cls, control_points: list[cp.Variable], opti: ca.Opti) -> "PBezierCurve":
        """
        Create a PBezierCurve instance from a list of control points.
        
        :param control_points: List of cvxpy variables representing the control points.
        :return: A new PBezierCurve instance with the given control points.
        """
        order                       = len(control_points) - 1
        bezier_curve                = PBezierCurve(order = order, 
                                                         dim   = control_points[0].shape[0], 
                                                         opti  = opti)
        bezier_curve.control_points = control_points
        
        return bezier_curve


    

class MinimumJerkPlannerCasadi:
    """
    A class to plan minimum jerk trajectories using Bezier curves.
    This class is a wrapper around the get_minimum_jerk_bezier_curve function.
    """

    def __init__(self, dim :int = 2, max_acceleration: float | None = None):
        
        """
            Initialize the MinimumJerkPlanner with the dimension of the space and an optional maximum acceleration constraint
            :param dim: Dimension of the space in which the Bezier curve will be defined.
            :param max_acceleration: Optional maximum acceleration constraint. If None, no constraint is applied
        """

        self.max_acceleration   = max_acceleration
        self.dim                = dim
        self.opti               = ca.Opti("conic")  # Create an Opti instance for optimization
        
        self.position_curve     = PBezierCurve(order=5, dim=dim, opti = self.opti)
        self.velocity_curve     = self.position_curve.get_derivative()
        self.acceleration_curve = self.velocity_curve.get_derivative()
        self.jerk_curve         = self.acceleration_curve.get_derivative()

        self.v0_par = self.opti.parameter(self.dim)
        self.v1_par = self.opti.parameter(self.dim)
        self.p0_par = self.opti.parameter(self.dim)
        self.p1_par = self.opti.parameter(self.dim)

        self.problem : cp.Problem = None
        self._setup()


    
    def _setup(self) :


        
        constraints = []
        # constraints for the initial and final velocities 
        constraints.append(self.velocity_curve.evaluate(0) == self.v0_par)
        constraints.append(self.velocity_curve.evaluate(1) == self.v1_par)
        # constraints for the initial and final positions
        constraints.append(self.position_curve.evaluate(0) == self.p0_par)
        constraints.append(self.position_curve.evaluate(1) == self.p1_par)

        if self.max_acceleration is not None:
            for points in self.acceleration_curve.control_points:
                constraints.append(ca.sumsqr(points) <= self.max_acceleration**2)

        # jerk cost along the lines
        cost   = 0
        t_span = np.linspace(0, 1, 1000)
        dt     = t_span[1] - t_span[0]  # Time step for numerical integration
        
        for t in t_span:
            jerk_value = self.jerk_curve.evaluate(t)
            cost += ca.sumsqr(jerk_value)*dt

        self.opti.subject_to(constraints)
        self.opti.minimize(cost)
        
        self.opti.solver("osqp")  # Set the solver for the optimization problem

        self.planner = self.opti.to_function('MPCPlanner',        [self.p0_par,self.p1_par, self.v0_par, self.v1_par],  [self.position_curve.stacked_control_points], 
                                                                  ['p0'      , 'p1'       ,'v0'        , 'v1']       ,  ['optimal_control_points',])
        
        
    
    
    def plan(self, p_0 : np.ndarray, p_1: np.ndarray,   v_0 : np.ndarray, v_1 : np.ndarray) -> tuple[BezierCurve, BezierCurve, BezierCurve, BezierCurve]:
        

        # set the parameters 
        
        control_points_array    = self.planner(p_0, p_1, v_0, v_1)  # Call the planner function with the parameters
        optimal_control_points  = [control_points_array[i*self.dim:(i+1)*self.dim].full().flatten() for i in range(self.position_curve.n)]  # Convert the output to a list of numpy arrays

        position_curve_opt     = BezierCurve(optimal_control_points)
        velocity_curve_opt     = position_curve_opt.get_derivative()
        acceleration_curve_opt = velocity_curve_opt.get_derivative()
        jerk_curve             = acceleration_curve_opt.get_derivative()

        return position_curve_opt, velocity_curve_opt, acceleration_curve_opt, jerk_curve
    
class STLSplinePlanner:
    """
    A class to plan STLSpline trajectories using Bezier curves.
    This class is a wrapper around the get_minimum_jerk_bezier_curve function.
    """

    def __init__(self, dim :int = 2, max_acceleration: float | None = None):
        """
            Initialize the STLSplinePlanner with the dimension of the space and an optional maximum acceleration constraint
            :param dim: Dimension of the space in which the Bezier curve will be defined.
            :param max_acceleration: Optional maximum acceleration constraint. If None, no constraint is applied
        """
        self.max_acceleration   = max_acceleration
        self.dim                = dim
        self.opti               = ca.Opti("conic")  # Create an Opti instance for optimization

        



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import perf_counter

    # print(factorial(11))  # Example usage of factorial function

    # fig, ax = plt.subplots(figsize=(8, 6))
    # # Example usage
    # control_points = [np.array([0, 0]), np.array([1, 2]), np.array([3, 0]), np.array([5, 2])]
    # bezier_curve = BezierCurve(control_points)
    # bezier_curve.draw(ax=ax, label='Bezier Curve', color='blue')

    # # get derivative 
    # fig, ax = plt.subplots(figsize=(8, 6))
    # derivative_curve = bezier_curve.get_derivative()
    # derivative_curve.draw(ax=ax, label='Derivative Bezier Curve', color='green')
    # plt.show()

    planner = MinimumJerkPlannerCasadi(dim=2)
    
    
    for jj in range(2):
        start = perf_counter()
        pos = planner.plan(np.array([0, 0]), np.array([5, 5]), np.array([0, -1]), np.array([0, 1]))
        end = perf_counter()
        print(f"Planning took {end - start:.4f} seconds.")

    
    # pos, vel, acc, jerk = get_minimum_jerk_bezier_curve(np.array([0, 0]), np.array([0, -1]), np.array([5, 5]), np.array([0, 1]), max_acceleration=3.0)
    pos.draw(label='Position Curve', color='blue')    
    # vel.draw(label='Velocity Curve', color='green')
    # acc.draw(label='Acceleration Curve', color='red')
    # jerk.draw(label='Jerk Curve', color='orange')
    plt.legend()
    plt.show()

   
