# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

from tensorflow import keras 
from tensorflow.keras.layers import PReLU, BatchNormalization, UpSampling2D, UpSampling3D, Conv2D, Conv3D, Add, Concatenate
from .dense_image_warp import dense_image_warp3d as warp

##############################################
#################### LOSSES ##################
##############################################
class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(y_pred.get_shape().as_list()) - 2
        vol_axes = list(range(1, ndims+1))

        top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
        bottom = tf.reduce_sum(y_true + y_pred, vol_axes)

        div_no_nan = tf.math.divide_no_nan if hasattr(tf.math, 'divide_no_nan') else tf.div_no_nan
        dice = tf.reduce_mean(div_no_nan(top, bottom))
        return -dice
    
class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            r = [d, *range(1, d), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad
    
##############################################
#################### LAYERS ##################
##############################################    
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

class CarSON():
    """Cardiac Segmentation Network."""
    
    def __init__(self, optimizer, opt):
        self.opt = opt
        self.optimizer = optimizer

    def compile_model(self, model):
        if not self.opt.isTrain:
            model.compile(loss=None, 
                          optimizer=self.optimizer(learning_rate=0))
        else:
            model.compile(loss=self.opt.criterion_netS, 
                          optimizer=self.optimizer(learning_rate=self.opt.netS_lr))

    def get_model(self):
        V = keras.Input(shape=self.opt.image_shape) 
        M = encoder_decoder(V, nchannels=self.opt.nlabels, map_activation='softmax')
        
        model = keras.Model(inputs=V, outputs=M)
        self.compile_model(model)
        
        return model
    
##############################################
################## NETWORKS ##################
############################################## 
class CarMEN():
    """Cardiac Motion Estimation Network."""
    
    def __init__(self, optimizer, opt):
        self.opt = opt
        self.optimizer = optimizer
        
    def compile_model(self, model, loss_w=None):
        if not self.opt.isTrain:
            model.compile(loss=None, 
                          optimizer=self.optimizer(learning_rate=0))
        else:
            model.compile(loss=self.opt.criterion_netME, 
                          loss_weights=loss_weights, 
                          optimizer=self.optimizer(learning_rate=self.opt.netME_lr))
    def get_model(self):    
        V_0 = keras.Input(shape=self.opt.volume_shape) 
        V_t = keras.Input(shape=self.opt.volume_shape)
        V   = keras.layers.Concatenate(axis=-1)([V_0, V_t])

        u = encoder_decoder(V, nchannels=3, map_activation=None)
                       
        if not self.opt.isTrain:
            model = keras.Model(inputs=[V_0, V_t], outputs=u)
            self.compile_model(model)
        else:
            inputs  = []
            outputs = []
            loss_w  = []
            if self.opt.lambda_i > 0.0:
                # 1. Intensity loss term
                V_0_pred = warp(V_t, motion_estimates)
                inputs  += [V_0, V_t]
                outputs += [V_0_pred]
                loss_w  += [self.opt.lambda_i]
            if self.opt.lambda_a > 0.0:
                # 2. Anatomical loss term
                M_0 = keras.Input(shape=self.opt.label_shape)
                M_t = keras.Input(shape=self.opt.label_shape)
                M_t_split = tf.split(M_t, M_t.shape[-1], -1)
                M_0_pred  = K.concatenate([warp(K.cast(mt, K.dtype(V_t)), motion_estimates) for mt in M_t_split], -1)    
                M_0_pred  = keras.activations.softmax(M_0_pred)                
                inputs  += [M_0, M_t]
                outputs += [M_0_pred]  
                loss_w  += [self.opt.lambda_a]
            if self.opt.lambda_s > 0.0:   
                # 3. Smoothness loss term adjusted by resolution
                res = keras.Input(shape=(1,1,1,3))
                inputs  += [res]
                outputs += [motion_estimates*res]  
                loss_w  += [self.opt.lambda_s]

            model = keras.Model(inputs=inputs, outputs=outputs)
            self.compile_model(model, loss_w=loss_w)
            
        return model