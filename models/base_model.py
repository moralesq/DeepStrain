# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod    
    def get_netS(self):
        """Generate and save segmentations."""
        pass
   
    @abstractmethod    
    def get_netME(self):
        """Generate and save motion estimates."""
        pass    
        
        