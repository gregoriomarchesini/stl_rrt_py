from typing import Union
import numpy as np

class TimeInterval :
    """time interval class"""
    _time_step = 1 # time step is used to switch time time to sampling instants
    # empty set is represented by a double a=None b = None
    def __init__(self,a : float =0. , b : float = 0.) -> None:
        
        a = float(a)
        b = float(b)
        if a > b :
            raise ValueError(f"Invalid interval [{a},{b}]. Time intervals must be such that a <= b.")
         
        self._a = a
        self._b = b
        
    @property
    def a(self) -> float:
        return self._a
    @property
    def b(self) -> float:
        return self._b
    
    @property
    def period(self) -> float:
        return self._b - self._a
    
    @property
    def aslist(self) -> list[float]:
        return [self._a,self._b]
    

    def is_singular(self) -> bool:
        a, b = self.a, self.b
        if a == b:
            return True
        else:
            return False

    
    def __eq__(self,time_int :  "TimeInterval") -> bool:
        """ equality check """
        a1,b1 = time_int.a,time_int.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return True
        else :
            return False
        
    def __ne__(self,time_int :  "TimeInterval") -> bool :
        """ inequality check """
        a1,b1 = time_int.a,time_int.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return False
        else :
            return True
        
    def __contains__(self,time_int: Union["TimeInterval",float]) -> bool :
        """subset relations self included in time_int  ::: time_int is subest of self """
        

        if isinstance(time_int,float) :
            if self._a <= time_int <= self._b :
                return True
            else :
                return False
        else :
            a1,b1 = time_int.a,time_int.b
            a2,b2 = self._a,self._b 

            if (a2<=a1) and (b2>=b1): # condition for intersectin without inclusion of two intervals
                return True
            else :
                return False
        
    def __truediv__(self,time_int: "TimeInterval") -> Union["TimeInterval", None] :
        """interval Intersection"""
        
        a1,b1 = time_int.a,time_int.b
        a2,b2 = self._a,self._b
        
        # the empty set is already in this cases since the empty set is included in any other set
        if self.is_right_of(time_int)  or self.is_left_of(time_int) :
            return None
        else :
            return TimeInterval(a = max(a1,a2), b = min(b1,b2))
        
                
    def is_right_of(self,time_int: "TimeInterval") -> bool :
        """check if self is left of time_int"""
        
        if (time_int.b <= self._a) :
            return True
        else :
            return False
        
    def is_left_of(self,time_int: "TimeInterval") -> bool :
        """check if self is right of time_int"""

        if (self._b <= time_int.a)  :
            return True
        else :
            return False
        
    def get_sample(self) -> float:
        """return the sample of the interval"""
        a,b = self._a,self._b
        if self.is_singular() :
            return a
        else :
            return a + np.random.rand() * (b-a)
        
    def __repr__(self)-> str:
        """string representation"""
        return f"TimeInterval({self._a},{self._b})"
    def __str__(self)-> str:
        """string representation"""
        return f"TimeInterval({self._a},{self._b})"



if __name__ == "__main__" :
    

    time_int = TimeInterval(0.3,2.1)
    time_int2 = TimeInterval(0.3,2)

    print(time_int in time_int2)