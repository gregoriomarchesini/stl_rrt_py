# class GEQ2d(Predicate):
#     """
#     Greater than predicate in the form x[0] - b >= 0
#     """
#     def __init__(self, dim :int,bound:float):
#         x_var = ca.MX.sym("x",2)
#         t_var = ca.MX.sym("t")

#         if dim not in [0,1]:
#             raise ValueError("The dimension must be 0 or 1")
#         self.bound = bound
        
#         predicate_function = ca.Function("greater_than",[x_var,t_var],[x_var[dim] - bound])
#         super().__init__(predicate_function)

# class LEQ2d(Predicate):
#     """
#     Less than predicate in the form x[0] - b >= 0
#     """
#     def __init__(self, dim :int,bound:float):
#         x_var = ca.MX.sym("x",2)
#         t_var = ca.MX.sym("t")
#         self.bound = bound

#         if dim not in [0,1]:
#             raise ValueError("The dimension must be 0 or 1")
        
#         predicate_function = ca.Function("less_than",[x_var,t_var],[bound - x_var[dim]])
#         super().__init__(predicate_function)

# class GEQ3d(Predicate):
#     """
#     Greater than predicate in the form x[0] - b >= 0
#     """
#     def __init__(self, dim :int,bound:float):
#         x_var = ca.MX.sym("x",3)
#         t_var = ca.MX.sym("t")
#         self.bound = bound

#         if dim not in [0,1,2]:
#             raise ValueError("The dimension must be 0, 1 or 2")
        
#         predicate_function = ca.Function("greater_than",[x_var,t_var],[x_var[dim] - bound])
#         super().__init__(predicate_function)

# class LEQ3d(Predicate):
#     """
#     Less than predicate in the form x[0] - b >= 0
#     """
#     def __init__(self, dim :int,bound:float):
#         x_var = ca.MX.sym("x",3)
#         t_var = ca.MX.sym("t")
#         self.b= bound

#         if dim not in [0,1,2]:
#             raise ValueError("The dimension must be 0, 1 or 2")
        
#         predicate_function = ca.Function("less_than",[x_var,t_var],[bound - x_var[dim]])
#         super().__init__(predicate_function)

# def box_predicates(lower_bound:list, upper_bound:list, return_predicates_list = False):
#     """
#     Create box predicate to have a bounded region in the state space
#     """
    
#     n_dims = len(lower_bound)

#     if len(lower_bound) != n_dims or len(upper_bound) != n_dims:
#         raise ValueError("The dimension of the bounds does not match the dimension of the state variable")
    
#     # create a subgle linear predicate for each bound in the form a_i x_i - b_i >=0
#     predicates = []
#     for i in range(n_dims):

#         if lower_bound[i] > upper_bound[i]:
#             raise ValueError("The lower bound must be less than or equal to the upper bound for each dimension. Mismatch found in dimension {}".format(i))
        
#         a = np.zeros(n_dims)
#         a[i] = 1.
#         b = lower_bound[i]
#         predicates.append(LinearPredicate(a,b))

#         a = np.zeros(n_dims)
#         a[i] = -1.
#         b = -upper_bound[i]
#         predicates.append(LinearPredicate(a,b))

    
#     new_formula = Formula(root = AndOperator(*predicates))
    
#     if return_predicates_list:
#         return new_formula, predicates
#     else :
#         return new_formula
    

# def box_predicate_2d(center:np.ndarray, width:float, return_predicates_list = False):
#     """
#     Create a box predicate in 2D
#     """
#     if len(center) != 2:
#         raise ValueError("The center must be a 2D vector")
    
#     lower_bound = center - width/2
#     upper_bound = center + width/2

#     return box_predicates(lower_bound, upper_bound, return_predicates_list)


# def plot_predicates(predicates: list[Predicate], x_min: float, x_max: float, y_min: float, y_max: float, n_points:int = 100):
#     """
#     Plots regions defined by the predicates in the state space.
#     """

#     fig, ax = plt.subplots()
#     x = np.linspace(x_min, x_max, n_points)
#     y = np.linspace(y_min, y_max, n_points)
    
#     BIG_NUMBER = 10000
#     X, Y = np.meshgrid(x, y)
#     levels = [-BIG_NUMBER , 0, BIG_NUMBER ]
    
    
#     Z = np.ones_like(X)*BIG_NUMBER 

#     for predicate in predicates:
#         for i in range(n_points):
#             for j in range(n_points):
#                 Z[i,j] = np.minimum(Z[i,j],np.asarray(predicate(np.array([X[i,j],Y[i,j]]),0)))
   
#     map = ax.contourf(X,Y,Z, levels = levels,cmap="binary")
    
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")

#     ax.set_xlim([x_min, x_max])
#     ax.set_ylim([y_min, y_max])

#     plt.colorbar(map, label=" Black (True) | White (False)")
#     plt.show()

