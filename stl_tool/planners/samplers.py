import numpy as np
from stl_tool.polyhedron import Polyhedron




class BiasedSampler:
    """
        This class is applied to create a biased sampler for STL planning.
        The sampler stores a lit of polytopes representing the predicate level set location of of 
        some STL formulas and the time at which these will have to be reached.

        At this point the sampler returns a sample in space time by first sampling a random time among the avilable times
        and then sampling a random point in the polytope corresponding to that time. 

        Thhis sampler is useful to steer the serach of trajectories towards the predicate level set locations of the STL formulas.
    """
    
    def __init__(self, list_of_polytopes : list[Polyhedron], list_of_times : list[float]):
        """
        
        :param list_of_polytopes: A list of Polyhedron objects representing the level set locations of the STL formulas.
        :type list_of_polytopes: list[Polyhedron]
        :param list_of_times: A list of times corresponding to each polytope in the list_of_polytopes.
        :type list_of_times: list[float]
        """
        
        
        self.list_of_polytopes :list[Polyhedron] = list_of_polytopes
        self.list_of_times     :list[float]      = list_of_times

        if len(list_of_polytopes) != len(list_of_times):
            raise ValueError("The number of polytopes and times must be the same.")
    
    def get_sample(self) -> np.ndarray :
        """
            This method samples a point in space-time from the stored polytopes and times.
            It randomly selects one of the polytopes and its corresponding time, then samples a point from that polytope.
        """
        
        

        # Choose randomly among them
        random_index = np.random.choice(len(self.list_of_polytopes))
        t_tilde  :float       = self.list_of_times[random_index]
        polytope :Polyhedron    = self.list_of_polytopes[random_index]
        x_tilde  :np.ndarray  = polytope.sample_random()
        
        return np.hstack((x_tilde.flatten(),t_tilde)) # return the sample in the form of (x,y,t)
    
class UnbiasedSampler:
    """
        This class is applied to create an unbiased sampler for STL planning.
        The sampler samples uniformly in the workspace and time.
        It is useful to explore the workspace without bias towards any specific region.
    """
    def __init__(self,workspace : Polyhedron):
        self.workspace = workspace

    def get_sample(self, max_time :float) -> np.ndarray :
        t_tilde  = np.random.uniform(0, max_time)
        x_tilde  :np.ndarray  = self.workspace.sample_random()
        return np.hstack((x_tilde.flatten(),t_tilde)) # return the sample in the form of (x,y,t)
          