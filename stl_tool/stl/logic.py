from typing import TypedDict
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import numpy as np


from ..polyhedron import Polyhedron
from .utils import TimeInterval

NodeID     = int
BIG_NUMBER = 1E8



##################################################
# De morgan laws
##################################################
#1) ¬ (φ1 ∧ φ2) = ¬φ1 ∨ ¬φ2
#2) ¬ (φ1 ∨ φ2) = ¬φ1 ∧ ¬φ2
#3) ¬ (G φ) = F (¬φ)
#4) ¬ (F φ) = G (¬φ)
#5)   G(φ1 ∧ φ2) = G φ1 ∧ G φ2
#6)   F(φ1 ∨ φ2) = F φ1 ∨ F φ2
#7)   F_{[a,b]}F_{[c,d]} φ = F_{[a+c,b+d]} φ
#8)   G_{[a,b]}G_{[c,d]} φ = G_{[a+c,b+d]} φ
#9)   distribute and over or : φ1 ∧ (φ2 ∨ φ3) = (φ1 ∧ φ2) ∨ (φ1 ∧ φ3) 
### Blocking Case
#    G(φ1 ∨ φ2) -> Blocking (no push forward rule)
#    F(φ1 ∧ φ2) -> Blocking (no push forward rule)

class Node :
    """
    Node of the formula tree. It contains the parent node and the node id.
    """
    __counter = 0
    def __init__(self):
        self.parent   : "Node" | None      = None
        self.children : list["Node"]       = []
        self.node_id  : NodeID             = Node.__counter
        Node.__counter += 1

class PredicateNode(Node):
    """
    Predicate Node of the formula tree. It stores the polytope defining the predicate and the dimensions of the state space that define such polytope.
    """
    def __init__(self, polytope: Polyhedron, output_matrix : np.ndarray, systems_id : list[int] = [], name:str | None = None ) -> None:
        """
        :param polytope: Polytope defining the predicate as Ax <= b
        :type polytope: Polytope
        :param output_matrix: Output matrix of the predicate. So the output is y = Cx and the predicate is defined as ACx <= b
        :type output_matrix: np.ndarray
        :param systems_id: List of system ids that the predicate is defined on. It is used to identify the system when multiple systems are used in the same formula.
        :type systems_id: list[int]
        :param name: name of the predicate
        :type name: str
        """

        self.parent               : "OperatorNode" |None    = None
        self.polytope             : Polyhedron              = polytope
        self.output_matrix        : np.ndarray              = output_matrix
        self.systems_id           : list[int]               = systems_id

        # check the output matrix
        # The polyhedron is defined as Ax <= b, but if you want the polyhedron to be defined on an output y = Cx then
        # the matrix C must be consistent with the matrix A
        

        if output_matrix.shape[0] != polytope.A.shape[1]:
            raise ValueError(f"The output matrix must have dimension compatible with the given predicate polyhedron such that the final predicate is given as ACx <= b. Given output matrix shape: {output_matrix.shape} and polyhedron A shape: {polytope.A.shape}")
        

        Node.__init__(self)  
        
        if name is None:
            self.name = "P_" + str(self.node_id)
        else:
            self.name = name
    

class OperatorNode(Node) :
    def __init__(self):
        raise NotImplementedError("Operator Nodes cannot be instantiated directly. Please use the subclasses of OperatorNode")


class AndOperator(OperatorNode) :
    """ And oerator node"""
    def __init__(self, *children : Node ) -> None:
        
        Node.__init__(self)
        if len(children) < 2:
            raise ValueError("The And operator must have at least two children")
        
    
        
        for child in children:
            if isinstance(child, PredicateNode):
                self.children += [child]
            elif isinstance(child, OperatorNode):
                self.children += [child]
            else:
                raise ValueError("The children of the And operator must be either predicates or other operators")

    @staticmethod     
    def are_all_predicates(nodes: list[Node]) -> bool:
        """
        Check if all children are predicates
        """
        for node in nodes:
            if not isinstance(node, PredicateNode):
                return False
        return True
    
class OrOperator(OperatorNode) :
    """ Or operator node """
    def __init__(self, *children : Node ) -> None:
         
        Node.__init__(self)
        if len(children) < 2:
            raise ValueError("The And operator must have at least two children")
        for child in children:
            if isinstance(child, PredicateNode):
                self.children += [child]
            elif isinstance(child, OperatorNode):
                self.children += [child]
            else:
                raise ValueError("The children of the And operator must be either predicates or other operators")
        
     
  
class GOp(OperatorNode) :
    """
    Always operator node. It is used to define the always operator in the formula tree in the form G(a,b).
    """
    def __init__(self, a = float, b = float):
        """
        Args:
            a (float): The beginning of the always operator
            b (float): The end of the always operator
        """

        Node.__init__(self)

        self.interval = TimeInterval(a,b)

    # attach temporal operator to formula
    def __rshift__(self, formula : "Formula") -> "Formula":
        
        """
        Attach temporal operator to formula. Push-forward rules are applied when possible
        """
        
        # rule  G_{[a,b]}G_{[c,d]} φ = G_{[a+c,b+d]} 
        if isinstance(formula.root, GOp):
            new_formula = Formula(root = GOp(self.interval.a + formula.root.interval.a, self.interval.b + formula.root.interval.b))
            new_formula.root.children = [formula.root.children[0]]
            return new_formula


        # rule G(φ1 ∧ φ2) = G φ1 ∧ G φ2
        elif isinstance(formula.root, AndOperator):
            new_child_formulas : list[Formula] = []
            for child in formula.root.children:
                child_formula       = Formula(root= child)
                child_new_formula   = GOp(self.interval.a, self.interval.b) >> child_formula # recurive call in order to get the nested formula until a temporal operator or predicate is reached
                new_child_formulas += [child_new_formula]
            
            new_formula = new_child_formulas[0]
            for i in range(1,len(new_child_formulas)):
                new_formula = new_formula & new_child_formulas[i]
            
            return new_formula
        
        
        # rule G(φ1 ∨ φ2) -> Blocking (no push forward rule)
        else :
            new_formula = Formula(root = GOp(self.interval.a, self.interval.b))
            new_formula.root.children = [formula.root]
            return new_formula


class FOp(OperatorNode) :
    """
    Eventually operator node. It is used to define the eventually operator in the formula tree in the form F(a,b).
    """
    def __init__(self, a = float, b = float):
        
        
        Node.__init__(self)

        self.interval = TimeInterval(a,b)
   

    def __rshift__(self, formula : "Formula") -> "Formula":
        
        """
        Attach temporal operator to formula. Push-forward rules are applied when possible.

        :param formula: Formula to attach the temporal operator to
        :type formula: Formula
        :return: New formula with the temporal operator attached
        :rtype: Formula
        
        Example:
            >>> f = Predicate(polytope, dims=[0,1], name="p")
            >>> g = FOp(0,10) >> f
        """
            

        # rule  F_{[a,b]}F_{[c,d]} φ = F_{[a+c,b+d]} 
        if isinstance(formula.root, FOp):
            new_formula = Formula(root = FOp(self.interval.a + formula.root.interval.a, self.interval.b + formula.root.interval.b))
            new_formula.root.children = [formula.root.children[0]]
            return new_formula

        #rule   F(φ1 ∨ φ2) = F φ1 ∨ F φ2
        elif isinstance(formula.root, AndOperator):
            new_child_formulas : list[Formula] = []
            for child in formula.root.children:
                child_formula       = Formula(root= child)
                child_new_formula   = FOp(self.interval.a, self.interval.b) >> child_formula # recurive call in order to get the nested formula until a temporal operator or predicate is reached
                new_child_formulas += [child_new_formula]
            
            new_formula = new_child_formulas[0]
            for i in range(1,len(new_child_formulas)):
                new_formula = new_formula | new_child_formulas[i]
            
            return new_formula
        
        #rule  F(φ1 ∧ φ2) -> Blocking (no push forward rule)
        else : 
            new_formula = Formula(root = FOp(self.interval.a, self.interval.b))
            new_formula.root.children = [formula.root]
            return new_formula


class NotOperator(OperatorNode) :
    def __init__(self,child : Node) -> None:
         
        Node.__init__(self)
        self.children = [child]

   
        

class UOp(OperatorNode):
    """
    Binary until operator node. It is used to define the until operator in the formula tree in the form U(a,b)
    """

    def __init__(self, a = float, b = float):
        
        Node.__init__(self)
        self.interval = TimeInterval(a,b)

        if a > b:
            raise ValueError("a must be less than b")
        if a<0 or b<0:
            raise ValueError("a and b must be positive")

        self.left_node  : Node = None 
        self.right_node : Node = None    



    def __rshift__(self, right_formula:"Formula") :
    
        if self.left_node is None:
            raise ValueError("The left formula must be defined before the right formula, The until operator is binary operator and it needs a left and right hand side formula")
        
        self.right_node = right_formula.root
        
        new_formula = Formula(root = self)
        new_formula.root.children = [self.left_node, self.right_node]
        return new_formula

        


def deepcopy_predicate_node(node: Node) -> Node:
    """ Go down the tree and duplicate the poredicates which coudl be shared among different branches"""

    if isinstance(node, PredicateNode):
        return PredicateNode(node.polytope, output_matrix= node.output_matrix, name= node.name)
    
    else:
        if isinstance(node,UOp):
            left_child = deepcopy_predicate_node(node.left_node)
            right_child = deepcopy_predicate_node(node.right_node)
            node.children = [left_child, right_child]
            return node
        
        else:
            new_children = []
            for child in node.children:
                new_children  += [deepcopy_predicate_node(child)]
            node.children = new_children
            
            return node
          
        
##############################################################################################################################################################
############### Formulas construction ########################################################################################################################
##############################################################################################################################################################

class Formula:
    """
    Abstract class to represent a Signal Temporal Logic Formula.
    It contains the root node of the formula tree and methods to manipulate the formula.
    """
    def __init__(self,root : Node | None = None) -> None:
        self._root = root # if None then it is an empty formula.


    @property
    def root(self):
        return self._root
    
    @property
    def is_null(self):
        """
        Check if the formula is null (i.e. it has no root)
        """
        return self._root is None
    

    def __and__(self, formula : "Formula"):
        """
        Defines conjunctions of formulas. It can be used to create a conjunction of formulas or to attach a formula to a conjunction of formulas.
        """


        # case 1: conjunction of two formulas with root as an AND operator.
        # sol  : create a unique and operator with the children of both formulas.

        if isinstance(formula.root, AndOperator)  and isinstance(self.root, AndOperator):

            root_self    = deepcopy_predicate_node(self.root)
            root_formula = deepcopy_predicate_node(formula.root)
            formula      = Formula(root = AndOperator(*root_self.children,*root_formula.children))
            
            return formula
           
        # case 2: conjunction of a formula with an AND operator as root for the other formula.
        # sol  : add the self formula as a child of the AND operator of the other formula.

        elif isinstance(formula.root, AndOperator) :
            root_self = deepcopy_predicate_node(self.root)
            root_formula = formula.root
            return Formula(root = AndOperator(root_self,*root_formula.children))
        
        # case 3: conjunction of a formula with an AND operator as root for the other formula.
        # sol  : add the formula as a child of the AND operator of the self formula.
        
        elif isinstance(self.root, AndOperator):
            root_formula = deepcopy_predicate_node(formula.root)
            root_self    = self.root
            return Formula(root = AndOperator(*root_self.children,root_formula))
        
        # empty formulas cases
        elif self.is_null :
            return formula
        
        elif formula.is_null :
            return self
        
        # case 4: conjunction of two formulas with root given by other operators 
        # sol  : create a new formula with the AND operator as root and the two formulas as children.
        else:
            root_self    = deepcopy_predicate_node(self.root)
            root_formula = deepcopy_predicate_node(formula.root)
            new_formula  = Formula( root = AndOperator(root_self ,root_formula))
            
            return new_formula
        
         
    # disjunction of formulas  
    def __or__(self, formula : "Formula"):
        """
        Overload the | operator to define disjunctions of formulas. 
        
        :param formula: Formula to disjoin with the current formula
        :type formula: Formula
        :return: New formula representing the disjunction of the two formulas
        :rtype: Formula

        Example:
            >>> f1 = Predicate(polytope1, dims=[0,1], name="p1")
            >>> f2 = Predicate(polytope2, dims=[0,1], name="p2")
            >>> f3 = f1 | f2
        """
        
        
        if isinstance(formula.root, OrOperator)  and isinstance(self.root, OrOperator):
            # (φ1 ∨ φ2) ∨ (φ3 ∨ φ4) = (φ1 ∨ φ2 ∨ φ3 ∨ φ4)

            root_self    = deepcopy_predicate_node(self.root)
            root_formula = deepcopy_predicate_node(formula.root)
            formula      = Formula(root = OrOperator(*root_self.children,*root_formula.children))
            
            return formula
        
        # distribute or operator
        elif isinstance(self.root, AndOperator) and isinstance(formula.root, OrOperator):
            # φ1 ∧ (φ2 ∨ φ3) = (φ1 ∧ φ2) ∨ (φ1 ∧ φ3) 
            # note: this operator can cause an exponential blow up of the formula size if used repeatedly
            
            new_child_formulas : list[Formula] = []
            for child in formula.root.children:
                child_formula       = Formula(root= child)
                child_new_formula   = self & child_formula # recurive call in order to get the nested formula until a temporal operator or predicate is reached
                new_child_formulas += [child_new_formula]
            
            new_formula = new_child_formulas[0]
            for i in range(1,len(new_child_formulas)):
                new_formula = new_formula | new_child_formulas[i]
            
            return new_formula
        
        elif isinstance(formula.root, AndOperator) and isinstance(self.root, OrOperator):
            # (φ1 ∨ φ2) ∧ φ3 = (φ1 ∧ φ3) ∨ (φ2 ∧ φ3) 
            # note: this operator can cause an exponential blow up of the formula size if used repeatedly

            new_child_formulas : list[Formula] = []
            for child in self.root.children:
                child_formula       = Formula(root= child)
                child_new_formula   = child_formula & formula # recurive call in order to get the nested formula until a temporal operator or predicate is reached
                new_child_formulas += [child_new_formula]
            
            new_formula = new_child_formulas[0]
            for i in range(1,len(new_child_formulas)):
                new_formula = new_formula | new_child_formulas[i]
            
            return new_formula
           
        elif isinstance(formula.root, OrOperator) :
            # (φ1 ∨ φ2) ∨ φ3 = (φ1 ∨ φ2 ∨ φ3)
            root_self     = deepcopy_predicate_node(self.root)
            root_formula   = formula.root
            return Formula(root = OrOperator(root_self,*root_formula.children))
        
        elif isinstance(self.root, OrOperator):
            # φ1 ∨ (φ2 ∨ φ3) = (φ1 ∨ φ2 ∨ φ3)
            root_formula = deepcopy_predicate_node(formula.root)
            root_self    = self.root
            return Formula(root = OrOperator(*root_self.children,root_formula))
        
        # empty formulas cases
        elif self.is_null:
            return formula
        
        elif formula.is_null:
            return self
        
        else:
            root_self    = deepcopy_predicate_node(self.root)
            root_formula = deepcopy_predicate_node(formula.root)
            new_formula  = Formula( root = OrOperator(root_self ,root_formula))
            
            return new_formula

        
    # negation of a formula
    def __invert__(self):

        #1) ¬ (φ1 ∧ φ2) = ¬φ1 ∨ ¬φ2
        #2) ¬ (φ1 ∨ φ2) = ¬φ1 ∧ ¬φ2
        #3) ¬ (G φ) = F (¬φ)
        #4) ¬ (F φ) = G (¬φ)
        #5) ¬ ¬ (φ) = φ
        #6) ¬ (φ1 U φ2) = G (¬φ1) ∨ (¬ φ2 U (¬ φ1 ∧ ¬ φ2)) # from: Robust Temporal Logic Model Predictive Control (Sadra Sadraddini and Calin Belta)

        if isinstance(self.root,AndOperator):
            
            child_formulas = []
            for child in self.root.children:
                child_formula = Formula(root = child)
                child_formula = ~ child_formula # will trigger the recursions
                child_formulas += [child_formula]
            

            child_roots = [child_formula.root for child_formula in child_formulas]
            new_formula = Formula(root = OrOperator(*child_roots))
            return new_formula
        
        elif isinstance(self.root,OrOperator):
            child_formulas = []
            for child in self.root.children:
                child_formula = Formula(root = child)
                child_formula = ~ child_formula
                child_formulas += [child_formula]


            child_roots = [child_formula.root for child_formula in child_formulas]
            new_formula = Formula(root = AndOperator(*child_roots))
            return new_formula
        
        elif isinstance(self.root, GOp):

            root_formula = Formula(root = self.root.children[0])
            return FOp(a = self.root.interval.a, b = self.root.interval.b) >> (~root_formula) 

        elif isinstance(self.root, FOp):
            root_formula = Formula(root = self.root.children[0])
            return GOp(a = self.root.interval.a, b = self.root.interval.b) >> (~root_formula)
        
        elif isinstance(self.root, UOp):

            a = self.root.interval.a
            b = self.root.interval.b

            left_formula  = Formula(root = self.root.children[0])
            right_formula = Formula(root = self.root.children[1])

            not_left_formula  = ~left_formula
            not_right_formula = ~right_formula
            new_formula = (GOp(a=a,b=b)>> not_left_formula ) | (not_right_formula << UOp(a=a,b=b) >> (not_left_formula & not_right_formula))

            return new_formula
            
        
        elif isinstance(self.root, PredicateNode):
            new_formula = Formula(root = NotOperator(self.root))
            return new_formula
        
        elif isinstance(self.root, NotOperator):
            # not-not removes the not
            new_formula = Formula(root = self.root.children[0])
            return new_formula
           
        # empty formulas cases
        elif self.is_null:
            return self
        
        else:
            raise ValueError("The formula must be either a predicate or an operator")

    def __lshift__(self, until_operator : "UOp") :

        if self.is_null :
            raise ValueError("The left formula must be defined before the right formula, The until operator is binary operator and it needs a left and right hand side formula which are not null formulas")

        until_operator.left_node = self.root
        return  until_operator # you return just an operator node which is not a formula! should give an error if combined with other formulas

        

    def has_temporal_operators(self):
        """
        Check if the formula has temporal operators
        """
        if self.is_null:
            return False

        def recursive_has_temporal_operators(node : Node):
            if isinstance(node, GOp) or isinstance(node, FOp) or isinstance(node, UOp):
                return True
            
            else :
                for child in node.children:
                    if recursive_has_temporal_operators(child):
                        return True
                    else:
                        return False
                return False # reached in case you have a predicate which has no children
        
        return recursive_has_temporal_operators(self.root)
    

    def max_horizon(self):
        """
        Get the horizon time of the formula
        """
        if self.is_null:
            return 0
        
        def recursive_horizon_time_of_formula(node : Node):

            # base case (predicate node):
            if isinstance(node, PredicateNode):
                return 0
            
            if isinstance(node, NotOperator):
                return recursive_horizon_time_of_formula(node.children[0])

            elif isinstance(node, GOp) or isinstance(node, FOp):
                return node.interval.b + recursive_horizon_time_of_formula(node.children[0])
            
            elif isinstance(node,UOp):
                phi1 = recursive_horizon_time_of_formula(node.children[0])
                phi2 = recursive_horizon_time_of_formula(node.children[1])
                return max(phi1, phi2) + node.interval.b

            elif isinstance(node, AndOperator) or isinstance(node, OrOperator):
                horizon_times = []
                for child in node.children:
                    horizon_times += [recursive_horizon_time_of_formula(child)]
                return max(horizon_times)
            
            else :
                raise ValueError("Unknown node type")

        return recursive_horizon_time_of_formula(self.root)

    def contains_disjuntion(self):
        """
        Check if the formula contains disjunctions
        """
        if self.is_null:
            return False
        
        def recursive_contains_disjunction(node):
            if isinstance(node, OrOperator):
                return True
            
            else :
                for child in node.children:
                    if recursive_contains_disjunction(child):
                        return True
                    else:
                        return False
                return False # reached in case you have a predicate which has no children
    
        
        return recursive_contains_disjunction(self.root)
    
    def formula_depth(self) -> int:
        
        if self.is_null:
            return 0
        
        recursive_formula_depth = lambda node: 1 + max([recursive_formula_depth(child) for child in node.children]) if len(node.children) else 1
        return recursive_formula_depth(self.root)
        
    def max_num_children(self) -> int:
        """
        Get the maximum number of children of the formula
        """
        if self.is_null:
            return 0

        def recursive_max_num_children(node):
            if len(node.children) == 0:
                return 0
            
            else :
                max_children = len(node.children)
                for child in node.children:
                    max_children = max(max_children, recursive_max_num_children(child))
                return max_children
        
        return recursive_max_num_children(self.root)

    def systems_in_the_formula(self) -> list[int]:
        """
        Get the list of systems in the formula
        """
        if self.is_null:
            return []
        
        systems = set()

        def recursive_systems_in_the_formula(node):
            if isinstance(node, PredicateNode):
                if node.systems_id is not None:
                    systems.update(node.systems_id)

            for child in node.children:
                recursive_systems_in_the_formula(child)
        
        recursive_systems_in_the_formula(self.root)
        
        return list(systems)
    

    def show_graph(self,debug=False):
    
        base_node_lateral_spacing = 6
        base_node_vertical_spacing = 2
        patch_radius = 0.2
        fig,ax = plt.subplots()

        # To track the bounds of the tree
        x_min, x_max, y_min, y_max = [0.,0.,0.,0.]
        temporal_color = "green"
        predicate_color = "yellow"
        logical_operator_color = "red"

        patch_specs = dict(edgecolor='black', facecolor = temporal_color, alpha=0.4, zorder=1, lw = 3)
        
        def plot_node(node: Node, x, y):
            if isinstance(node, PredicateNode):
                if debug:
                    ax.text(x, y, f"P\nnode_id: {node.node_id}", fontsize=12, ha='center', va='center')
                else:
                    ax.text(x, y, "P", fontsize=12, ha='center', va='center')
                Rect = patches.Rectangle((x-patch_radius, y-patch_radius), width=2*patch_radius, height=2*patch_radius, **patch_specs)
                ax.add_patch(Rect)

            elif isinstance(node, AndOperator):
                if debug:
                    ax.text(x, y, f"AND\nnode_id: {node.node_id}", fontsize=12, ha='center', va='center')
                else:
                    ax.text(x, y, "AND", fontsize=12, ha='center', va='center')
                circle = patches.Circle((x, y), radius=patch_radius, **patch_specs)
                ax.add_patch(circle)

            elif isinstance(node, UOp):
                label = f"U_[{node.interval.a},{node.interval.b}]"
                if debug:
                    ax.text(x, y, f"{label}\nnode_id: {node.node_id}", fontsize=12, ha='center', va='center')
                else:
                    ax.text(x, y, label, fontsize=12, ha='center', va='center')
                circle = patches.Circle((x, y), radius=patch_radius, **patch_specs)
                ax.add_patch(circle)

            elif isinstance(node, OrOperator):
                if debug:
                    ax.text(x, y, f"OR\nnode_id: {node.node_id}", fontsize=12, ha='center', va='center')
                else:
                    ax.text(x, y, "OR", fontsize=12, ha='center', va='center')
                circle = patches.Circle((x, y), radius=patch_radius, **patch_specs)
                ax.add_patch(circle)

            elif isinstance(node, GOp):
                label = f"G_[{node.interval.a},{node.interval.b}]"
                if debug:
                    ax.text(x, y, f"{label}\nnode_id: {node.node_id}", fontsize=12, ha='center', va='center')
                else:
                    ax.text(x, y, label, fontsize=12, ha='center', va='center')
                circle = patches.Circle((x, y), radius=patch_radius, **patch_specs)
                ax.add_patch(circle)

            elif isinstance(node, FOp):
                label = f"F_[{node.interval.a},{node.interval.b}]"
                if debug:
                    ax.text(x, y, f"{label}\nnode_id: {node.node_id}", fontsize=12, ha='center', va='center')
                else:
                    ax.text(x, y, label, fontsize=12, ha='center', va='center')
                circle = patches.Circle((x, y), radius=patch_radius, **patch_specs)
                ax.add_patch(circle)

            elif isinstance(node, NotOperator):
                if debug:
                    ax.text(x, y, f"NOT\nnode_id: {node.node_id}", fontsize=12, ha='center', va='center')
                else:
                    ax.text(x, y, "NOT", fontsize=12, ha='center', va='center')
                circle = patches.Circle((x, y), radius=patch_radius, **patch_specs)
                ax.add_patch(circle)

            else:
                raise ValueError("Unknown node type")
                    
            # Track the x and y limits
            nonlocal x_min, x_max, y_min, y_max
            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)

        plot_node(self.root, 0, 0)

        def plot_tree(node:Node, x, y,level = 0):
            # Plot the current node
            plot_node(node, x, y)
            
            # Recurse on children
            if len(node.children) >=2: #(binary operator)
                n_children = len(node.children)
                for i, child in enumerate(node.children):
                    # Calculate the child position
                    child_x = x + (i - (n_children - 1) / 2) * base_node_lateral_spacing/(2*level+1)
                    child_y = y - base_node_vertical_spacing

                    # Draw a line from current node to child
                    ax.plot([x, child_x], [y-patch_radius, child_y+patch_radius], 'k-', lw=1)

                    # Recursive call to plot the subtree
                    plot_tree(child, child_x, child_y, level+1)
            
            elif len(node.children) ==1 : # unary operator
                child = node.children[0]
                child_x = x
                child_y = y - base_node_vertical_spacing
                ax.plot([x, child_x], [y-patch_radius, child_y+patch_radius], 'k-', lw=1)
                plot_tree(child, child_x, child_y, level+1)

            else: #(predicate nodes do not have children. Nothing to do)
                pass


        plot_tree(self.root, 0, 0)
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])

    

    def enumerate_predicates(self) -> dict[str,Polyhedron] :
        """
        Get the predicates of the formula
        """

        if self.is_null:
            return {}
    
        def recursive_get_predicates(node: Node):
            if isinstance(node, PredicateNode):
                return {node.name: node.polytope}
            else:
                predicates = {}
                for child in node.children:
                    predicates.update(recursive_get_predicates(child))
                return predicates

        return recursive_get_predicates(self.root)

        
class Predicate(Formula) :
    """
    Wrapper class to create a predicate formula from a polytope.
    """
    def __init__(self, polytope: Polyhedron, output_matrix : np.ndarray , systems_id: list[int] = [], name: str | None = None) -> None:
        """
        
        :param polytope: Polytope defining the predicate
        :type polytope: Polytope
        :param dims: dimensions of the state space that define the predicate. These are equal to the dimensions of the polytope.
        :type dims: list[int]
        :param name: name of the predicate
        :type name: str
        """
        super().__init__(root = PredicateNode(polytope = polytope , output_matrix = output_matrix , systems_id = systems_id , name = name))

    @property
    def polytope(self) -> Polyhedron:
        """
        Get the polytope of the predicate node
        """
        return self.root.polytope
    @property
    def output_matrix(self) -> np.ndarray:
        """
        Get the output matrix of the predicate node
        """
        return self.root.output_matrix
    
    @property
    def name(self) -> str:
        """
        Get the name of the predicate node
        """
        return self.root.name
    
    # def __add__(self, other:"Predicate") -> "Predicate" :
    #     """
    #     Cartesian product of two polytopes using the '+' operator.
        
    #     :param other: Another Polytope object to combine with.
    #     :type other: Polytope
    #     :return: A new Polytope object representing the Cartesian product.
    #     :rtype: Polytope
    #     """

    #     if not isinstance(other, Predicate):
    #         raise ValueError(f"Cannot add {type(other)} to Predicate. Only Predicate is supported.")
        
    #     old_polytope = self.polytope
    #     new_polytope = old_polytope.cross(other.polytope)

    #     # shift dimensions of the other polytope
    #     new_dims = self.dims + [d + len(self.dims) for d in other.dims]

    #     name = f"{self.name} + {other.name}" if self.name and other.name else None
    #     return Predicate(polytope=new_polytope, dims=new_dims, name=name)

    # def __radd__(self, other:"Predicate") -> "Predicate" :
    #     """
    #     Cartesian product of two polytopes using the '+' operator.
        
    #     :param other: Another Polytope object to combine with.
    #     :type other: Polytope
    #     :return: A new Polytope object representing the Cartesian product.
    #     :rtype: Polytope
    #     """

    #     if not isinstance(other, Predicate):
    #         raise ValueError(f"Cannot add {type(other)} to BoxBound. Only BoxBound is supported.")
        
    #     old_polytope = other.polytope
    #     new_polytope = old_polytope.cross(self.polytope)
    #     # shift dimensions of self
    #     new_dims = other.dims + [d + len(other.dims) for d in self.dims]
    #     name = f"{other.name} + {self.name}" if self.name and other.name else None
    #     return Predicate(polytope=new_polytope, dims=new_dims, name=name)

def get_fomula_type_and_predicate_node(formula : Formula ) -> tuple[str,PredicateNode] :

    """
    This function takes a fomula anch check if it is a G, F, GF or FG operator. If it is not of these types then it resturns an error.

    :param formula: Formula to check
    :type formula: Formula
    :return: Tuple with the type of the formula and the predicate node
    :rtype: tuple[str,PredicateNode]
    """

    if not isinstance(formula,Formula) : 
        raise ValueError("The formula must be of type Formula")
 
    root_node : Node = formula.root
    if isinstance(root_node,GOp):
        if isinstance(root_node.children[0],PredicateNode):
            predicate_node = root_node.children[0]
            return "G", predicate_node
    
    if isinstance(root_node,FOp):
        if isinstance(root_node.children[0],PredicateNode):
            predicate_node = root_node.children[0]
            return "F", predicate_node
        
    
    if isinstance(root_node,GOp):
        if isinstance(root_node.children[0],FOp):
            if isinstance(root_node.children[0].children[0],PredicateNode):
                predicate_node = root_node.children[0].children[0]
                return "GF", predicate_node
    
    if isinstance(root_node,FOp):
        if isinstance(root_node.children[0],GOp):
            if isinstance(root_node.children[0].children[0],PredicateNode):
                predicate_node = root_node.children[0].children[0]
                return "FG", predicate_node
    
    raise ValueError("The given formula is either a conjunction, a disjunction or a a formulaa containing the Until operator. Please provide a formula of type Gp, Fp, GFp or FGp where p is the predicate node.")  


# if __name__ == "__main__":
    
    
#     A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
#     b = np.array([1, 1, 1, 1])
#     polytope = Polytope(A, b)

#     # Create a predicate node
#     predicate_node = Predicate(polytope)
#     formula        = (~((((GOp(a=0, b=1) >> predicate_node) |  (GOp(a=0, b=1) >> (~predicate_node) ))) << UOp(1,2) >>  ((GOp(a=0, b=1) >> predicate_node) &  (GOp(a=0, b=1) >> (~predicate_node) )))) & (~((((GOp(a=0, b=1) >> predicate_node) |  (GOp(a=0, b=1) >> (~predicate_node) ))) << UOp(1,2) >>  ((GOp(a=0, b=1) >> predicate_node) &  (GOp(a=0, b=1) >> (~predicate_node) ))))

#     stl_graph(formula,debug=    True)

#     # Show the animation
#     plt.tight_layout()
#     plt.show()