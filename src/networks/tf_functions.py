import tensorflow as tf
from tensorflow.keras.layers import *

def conv(Conv, layer_input, filters, kernel_size=3, strides=1, residual=False):
    """Convolution layer: Ck=Convolution-BatchNorm-PReLU"""
    dr = Conv(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
    d  = BatchNormalization(momentum=0.5)(dr)  
    d  = PReLU()(d)
    
    if residual:
        return dr, d
    else:
        return d

def deconv(Conv, UpSampling, layer_input, filters, kernel_size=3, strides=1):
    """Deconvolution layer: CDk=Upsampling-Convolution-BatchNorm-PReLU"""
    u = UpSampling(size=strides)(layer_input)
    u = conv(Conv, u, filters, kernel_size=kernel_size, strides=1)
    return u
    
def encoder(Conv, layer_input, filters, kernel_size=3, strides=2):
    """Layers for 2D/3D network used during downsampling: CD=Convolution-BatchNorm-LeakyReLU"""
    d = conv(Conv, layer_input, filters, kernel_size=kernel_size, strides=1)
    dr, d = conv(Conv, d, filters, kernel_size=kernel_size, strides=strides, residual=True)
    d  = Conv(filters, kernel_size=kernel_size, strides=1, padding='same')(d)
    d  = Add()([dr, d])
    return d

def decoder(Conv, UpSampling, layer_input, skip_input, filters, kernel_size=3, strides=2):
    """Layers for 2D/3D network used during upsampling"""
    u = conv(Conv, layer_input, filters, kernel_size=1, strides=1)
    u = deconv(Conv, UpSampling, u, filters, kernel_size=kernel_size, strides=strides)
    u = Concatenate()([u, skip_input])
    u = conv(Conv, u, filters, kernel_size=kernel_size, strides=1)
    return u

def encoder_decoder(x, gf=64, nchannels=3, map_activation=None):
    
    if len(x.shape) == 5:
        Conv        = Conv3D
        UpSampling  = UpSampling3D
        strides     = (2,2,1)
        kernel_size = (3,3,1)
    elif len(x.shape) == 4:
        Conv        = Conv2D
        UpSampling  = UpSampling2D
        strides     = (2,2)
        kernel_size = (3,3)
            
    d1 = encoder(Conv, x,  gf*1, strides=strides, kernel_size=kernel_size)
    d2 = encoder(Conv, d1, gf*2, strides=strides, kernel_size=kernel_size)
    d3 = encoder(Conv, d2, gf*4, strides=strides, kernel_size=kernel_size)
    d4 = encoder(Conv, d3, gf*8, strides=strides, kernel_size=kernel_size)
    d5 = encoder(Conv, d4, gf*8, strides=strides, kernel_size=kernel_size)
    d6 = encoder(Conv, d5, gf*8, strides=strides, kernel_size=kernel_size)
    d7 = encoder(Conv, d6, gf*8, strides=strides, kernel_size=kernel_size)
    
    u1 = decoder(Conv, UpSampling, d7, d6, gf*8, strides=strides, kernel_size=kernel_size)
    u2 = decoder(Conv, UpSampling, u1, d5, gf*8, strides=strides, kernel_size=kernel_size)
    u3 = decoder(Conv, UpSampling, u2, d4, gf*8, strides=strides, kernel_size=kernel_size)
    u4 = decoder(Conv, UpSampling, u3, d3, gf*4, strides=strides, kernel_size=kernel_size)
    u5 = decoder(Conv, UpSampling, u4, d2, gf*2, strides=strides, kernel_size=kernel_size)
    u6 = decoder(Conv, UpSampling, u5, d1, gf*1, strides=strides, kernel_size=kernel_size)

    u7 = UpSampling(size=strides)(u6)
    u7 = Conv(nchannels, kernel_size=kernel_size, strides=1, padding='same', activation=map_activation)(u7)    
    
    return u7