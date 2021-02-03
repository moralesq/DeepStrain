from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod    
    def get_netS(self):
        """Generate and save segmentations."""
        pass
   
    @abstractmethod    
    def get_netD(self):
        """Generate and save motion estimates."""
        pass    
        
        