import random
import numpy as np
import  matplotlib.pyplot as plt
from json import loads


from   ..polyhedron import Box2d, Box3d, Polyhedron,selection_matrix_from_dims
from   ..stl.logic import PredicateNode, Node
from   ..stl.logic import Formula 


class Map:
    """
    Helper class to define a simple 2d map with obstacles
    """

    def __init__(self, workspace : Polyhedron) -> None:
        """
        Args:
            bounds : list of tuples
                each tuple represents the min and max along each dimension
        """

        self.workspace   : Polyhedron              = workspace
        self.obstacles   : list[Polyhedron]        = []
        self.obstacles_inflated : list[Polyhedron] = []
        self.ax          : plt.Axes              = None
        self.fig         : plt.Figure            = None

    
    def _add_obstacle(self, obstacle: Polyhedron ) -> None: 

        """
        Adds an obstacle to the map
        Args:
            obstacle : Box2d or list of Box2d
                obstacle to add
        """
      
        if obstacle.num_dimensions < self.workspace.num_dimensions:
            print("obstacle and workspace must have the same number of dimensions. Obstacles dimensional inflating")
            # inflate the obstacle to the workspace dimension
            obstacle : Polyhedron = obstacle.get_inflated_polytope_to_dimension(self.workspace.num_dimensions)
            self.obstacles.append(obstacle)
            self.obstacles_inflated.append(obstacle)
        
        elif obstacle.num_dimensions == self.workspace.num_dimensions:
            self.obstacles.append(obstacle)
            self.obstacles_inflated.append(obstacle)
        
        else: 
            raise ValueError("Obstacle must be a Polytope dimension lower or equal to the workspace dimension. Given : {}".format(obstacle.num_dimensions))
               
    
    def add_obstacle(self, obstacle: Polyhedron |list[Polyhedron] ) -> None:

        """
        Adds an obstacle to the map
        Args:
            obstacle : Box2d or list of Box2d
                obstacle to add
        """
        if isinstance(obstacle, Polyhedron):
            self._add_obstacle(obstacle)
        elif isinstance(obstacle, list):
            for obs in obstacle:
                self._add_obstacle(obs)
        else:
            raise ValueError("obstacle must be a Polytope or list of Polytope. Given : {}".format(type(obstacle)))

    def enlarge_obstacle(self,border_size:float = 0.1) -> None:
        
        self.obstacles_inflated = []
        for obstacle in self.obstacles:
            # inflate the obstacle to the workspace dimension
            v = np.ones(obstacle.num_hyperplanes)
            # inflate the obstacle by border_size
            self.obstacles_inflated.append(Polyhedron(A = obstacle.A, b = obstacle.b + border_size*v))


    def draw(self, ax = None , projection_dim: list[int] = [], alpha: float = 1.) :

        if len(projection_dim) == 0:  
            # just project the first available dimensions 
            projection_dim = [i for i in range(min(3,self.workspace.num_dimensions))]
        
        elif not len(projection_dim) in [2,3] : 
            raise ValueError("projection_dim must have length 2 or 3. Given list is {}".format(projection_dim))
    
        # make appropriate projections
        if  len(projection_dim) == 2:
            if ax is not None:
                fig = ax.figure
            else:
                fig, ax = plt.subplots(figsize=(10, 10))
            # draw black contour of the workspace using vertices
            vertices = self.workspace.projection(projection_dim).vertices
            # add first vertex in the end to close the loop
            vertices = np.concatenate((vertices, vertices[0:1]), axis=0)
            ax.plot(vertices[:, 0], vertices[:, 1], 'k', linewidth=3)

        elif len(projection_dim) == 3:
            if ax is not None:
                if not hasattr(ax, "get_proj"):
                    print("Make new axis because the plot is 3d but the gives axis is not")
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection='3d')
                else:
                    fig = ax.figure
            else:
                #make 3d plot
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

            # draw black contour of the workspace using vertices
            vertices = self.workspace.projection(projection_dim).vertices
            # add first vertex in the end to close the loop
            vertices = np.concatenate((vertices, vertices[0:1]), axis=0)
            ax.plot(vertices[:, 0], vertices[:, 1],vertices[:, 2], 'k', linewidth=0.01) # it will just be invisible but set the correct size for the map
        
        # draw obstacles
        for obstacle in self.obstacles:
            obstacle.plot(ax, color='k', alpha=alpha, projection_dims = projection_dim)
        
        self.fig = fig
        self.ax  = ax

        return fig,ax
    
    def draw_formula_predicate(self,formula : Formula, projection_dim: list[int] = [], alpha: float = 1., with_label : bool = False) -> tuple[plt.Figure, plt.Axes]:

        if len(projection_dim) == 0:  
            # just project the first available dimensions 
            projection_dim = [i for i in range(min(3,self.workspace.num_dimensions))]
        
        if self.ax is None:
            self.fig, self.ax = self.draw(projection_dim)

        # look recursuvely into the tree 
        def draw_recursively(node: Node, with_label = False):
            if isinstance(node, PredicateNode):
                if node.polyhedron.num_dimensions != self.workspace.num_dimensions :
                    try :
                        C = selection_matrix_from_dims(n_dims = self.workspace.num_dimensions, selected_dims = node.dims)
                    except IndexError as e:
                        raise IndexError("There was a problem plotting the predicate. The main cause is probably due to the an inconsistenty " \
                              "between the workspace dimension and the selected output indices of one of the predicates. For example you  created" \
                              " a predicate over dims =[1,2] but the workspace has only dimension 2, such that dimension 2 is out of bounds. " \
                              "The given exception is : {}".format(e))
                    polytope = Polyhedron(node.polyhedron.A@C, b = node.polyhedron.b) # bring polytope to suitable dimension
                else :
                    polytope = node.polyhedron
                polytope.plot(self.ax, alpha=alpha,color='b',projection_dims= projection_dim)
                # plot the name 
                if node.name is not None and not polytope.is_open and with_label:
                    vertices = polytope.projection(projection_dim).vertices
                    center = np.mean(vertices, axis=0)
                    if len(projection_dim) == 2:
                        self.ax.text(center[0], center[1], node.name, fontsize=8)
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

        point = point.flatten()
        point = point[:max(3,len(point))]
        if hasattr(self.ax, "get_proj") and len(point)<3:
            point = np.stack((point,0),axis=0) # add a third dimension if not 3d
        elif not hasattr(self.ax, "get_proj") :
            point = point[:2]
        
        
        if not hasattr(self.ax, "get_proj"): # 2d plot
            self.ax.scatter(point[0], point[1], color=color, label=label)
        else: # 3d plot
            self.ax.scatter(point[0], point[1], point[2], color=color, label=label)
        
        return self.fig,self.ax

    def read_from_json(self, json_file: str):
        """
        Read map from json file
        Args:
            json_file : str
                path to json file
        """
        
        with open(json_file, "r") as f:
            map_json = loads(f.read())

            
        for object in map_json:
            if object["name"].split("_")[0] == "obstacle":
                if "center_z" in object.keys():
                    center = np.array([object["center_x"], object["center_y"], object["center_z"]])
                    size   = np.array([object["size_x"], object["size_y"], object["size_z"]])
                    self.add_obstacle(Box3d(x = center[0],y = center[1],z=center[2],size = size))
                else:
                    center = np.array([object["center_x"], object["center_y"]])
                    size   = np.array([object["size_x"], object["size_y"]])
                    self.add_obstacle(Box2d(x = center[0],y = center[1],size = size))
                
    

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