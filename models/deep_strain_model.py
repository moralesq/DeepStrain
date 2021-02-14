# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

from .base_model import BaseModel
from . import networks

class DeepStrain(BaseModel):
    
    def __init__(self, optimizer, opt):
        BaseModel.__init__(self, opt)
        self.optimizer = optimizer
           
    def get_netS(self):
        carson = networks.CarSON(self.optimizer, self.opt)
        netS = carson.get_model()
        netS.load_weights(self.opt.pretrained_models_netS)
        return netS
       
    def get_netME(self):
        carmen = networks.CarMEN(self.optimizer, self.opt)
        netME = carmen.get_model()
        netME.load_weights(self.opt.pretrained_models_netME)
        return netME