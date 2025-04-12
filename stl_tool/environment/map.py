import random
import numpy as np
import  matplotlib.pyplot as plt


from   stl_tool.polytope import Box2d, Box3d, Polytope
from   stl_tool.stl.logic import PredicateNode, Node
from   stl_tool.stl.logic import Formula 



class Map:
    """
    Helper class to define a simple 2d map with obstacles
    """

    def __init__(self, workspace : Polytope) -> None:
        """
        Args:
            bounds : list of tuples
                each tuple represents the min and max along each dimension
        """

        self.workspace   :Polytope        = workspace
        self.obstacles   : list[Polytope] = []
        self.ax          : plt.Axes       = None
        self.fig         : plt.Figure     = None
    
    def add_obstacle(self, obstacle: Polytope | list[Polytope]) -> None: 

        """
        Adds an obstacle to the map
        Args:
            obstacle : Box2d or list of Box2d
                obstacle to add
        """
        if isinstance(obstacle, Polytope):
            if obstacle.num_dimensions != self.workspace.num_dimensions:
                raise ValueError("obstacle and workspace must have the same number of dimensions")
            self.obstacles.append(obstacle)
            
        elif isinstance(obstacle, list):
            for obs in obstacle:
                if isinstance(obs, Polytope):
                    if obs.num_dimensions != self.workspace.num_dimensions:
                        raise ValueError("obstacle and workspace must have the same number of dimensions")
                    self.obstacles.append(obs)
                else:
                    raise ValueError("obstacle must be a Polytope or list of Box2d")
        else:
            raise ValueError("obstacle must be a Box2d or list of Box2d. Given : {}".format(type(obstacle)))
    

    def draw(self, list_of_dimensions: list[int] = []) :
        

        
        if not len(list_of_dimensions) and self.workspace.num_dimensions >3:
            raise ValueError("list_of_dimensions must be provided for 4D or higher dimensions. PLotting can only be done for 3 or 2 dimension")
        elif self.workspace.num_dimensions == 1:
            raise ValueError("1D map cannot be plotted yet")
        else : #otherwise plor normally
            if len(list_of_dimensions) :
                projected_workspace = self.workspace.projection(list_of_dimensions)
                projected_obstacles = [obstacle.projection(list_of_dimensions) for obstacle in self.obstacles]
            else : # nothing to project if the workspace is already in a plottable dimension
                projected_workspace = self.workspace
                projected_obstacles = self.obstacles

        if projected_workspace.num_dimensions == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            # draw black contour of the workspace using vertices
            vertices = projected_workspace.vertices
            # add first vertex in the end to close the loop
            vertices = np.concatenate((vertices, vertices[0:1]), axis=0)
            ax.plot(vertices[:, 0], vertices[:, 1], 'k', linewidth=3)
            # draw obstacles
            for obstacle in projected_obstacles:
                obstacle.plot(ax, color='k', alpha=0.3)

    
        elif projected_workspace.num_dimensions == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            # draw black contour of the workspace using vertices
            vertices = projected_workspace.vertices
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'k', linewidth=0.0001) # just don't make them visible
            # draw obstacles
            for obstacle in projected_obstacles:
                obstacle.plot(ax, color='k', alpha=0.3)
        
        self.fig = fig
        self.ax  = ax

        return fig,ax
    
    def draw_formula_predicate(self,formula : Formula):
        
        if self.ax is None:
            self.fig, self.ax = self.draw()

        # look recursuvely into the tree 
        def draw_recursively(node: Node):
            if isinstance(node, PredicateNode):
                node.polytope.plot(self.ax, alpha=0.3,color='b')
            else:
                for child in node.children:
                    draw_recursively(child)
        
        draw_recursively(formula.root)

        return self.fig,self.ax
            
    def show_point(self, point: np.ndarray, color = 'r', label = None):
        """
        Show a point in the map
        Args:
            point : np.ndarray
                point to show
            color : str
                color of the point
            label : str
                label of the point
        """
        if self.ax is None:
            self.fig, self.ax = self.draw()
        
        if self.workspace.num_dimensions == 2:
            self.ax.scatter(point[0], point[1], color=color, label=label)
        elif self.workspace.num_dimensions == 3:
            self.ax.scatter(point[0], point[1], point[2], color=color, label=label)
        else: # plot first three dimensions in case
            self.ax.scatter(point[0], point[1], point[2], color=color, label=label)
        
        return self.fig,self.ax




if __name__ == "__main__":
    # obstacles
    obstacles = [
        Box3d(3, 3, 3, size = [5, 1, 1]),
        Box3d(5, 5, 3, size = [2, 2, 1]),
        Box3d(7, 7, 3, size = [1, 1, 1]),
    ]

    workspace = Box3d(0, 0, 0, size = [20, 20,20])

    # create map
    map = Map(workspace)
    map.add_obstacle(obstacles)

    # draw
    map.draw()
    plt.show()