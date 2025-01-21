import numpy as np

class Operators:
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1
    
    def lf(x): 
        return 1 / (1 + np.exp(-x))
    