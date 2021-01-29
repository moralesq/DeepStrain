
import tensorflow.keras as keras
from .tf_functions import encoder_decoder

class CarSON():
    
    def __init__(self, image_shape=(128,128,1), label_shape=(128,128,4), criterion='categorical_crossentropy'):
        """Cardiac Segmentation Network (CarSON)"""
        
        self.name = None
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.nlabels     = self.label_shape[-1]
        self.criterion   = criterion
        
    def initialize_model(self):
            
        V = keras.Input(shape=self.image_shape) 
    
        segmentation = encoder_decoder(V, nchannels=self.nlabels, map_activation='softmax')
        self.model = keras.Model(inputs=V, outputs=segmentation)
        
        self.model.compile(loss=self.criterion, optimizer=keras.optimizers.Adam(learning_rate=0.0005))
