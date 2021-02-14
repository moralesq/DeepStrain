import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.measurements import center_of_mass

class MyocardialStrain():
    
    def __init__(self, mask, flow):
                
        self.mask  = mask
        self.flow  = flow
        
        assert len(mask.shape) == 3
        assert len(flow.shape) == 4
        assert mask.shape == flow.shape[:3]
        assert flow.shape[-1] == 3
        
    def calculate_strain(self, dx=1, dy=1, dz=1, lv_label=3):
        
        cx, cy, cz = center_of_mass(self.mask==lv_label)
        nx, ny, nz = self.mask.shape
        
        self.flow_rot = _roll_to_center(self.flow, cx, cy)
        self.mask_rot = _roll_to_center(self.mask, cx, cy)

        ux, uy, uz  = np.array_split(self.flow_rot, 3, -1)
        Uxx,Uxy,Uxz = np.gradient(np.squeeze(ux),dx,dy,dz)
        Uyx,Uyy,Uyz = np.gradient(np.squeeze(uy),dx,dy,dz)
        Uzx,Uzy,Uzz = np.gradient(np.squeeze(uz),dx,dy,dz)

        self.E_cart = np.zeros((nx,ny,nz,3,3))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Ugrad = [[Uxx[i,j,k], Uxy[i,j,k], Uxz[i,j,k]], 
                             [Uyx[i,j,k], Uyy[i,j,k], Uyz[i,j,k]],
                             [Uzx[i,j,k], Uzy[i,j,k], Uzz[i,j,k]]]
                    F = np.array(Ugrad) + np.identity(3)
                    e = 0.5*(np.matmul(F.T, F) - np.identity(3))
                    self.E_cart[i,j,k] += e

        self.Ezz = self.E_cart[:,:,:,2,2]
        self.Err, self.Ecc = self._convert_to_polar(self.E_cart[:,:,:,:2,:2])

        
        
        
    def _convert_to_polar(self, E):

        phi = _polar_grid(*E.shape[:2])[0]
        Err = np.zeros(self.mask.shape)
        Ecc = np.zeros(self.mask.shape)
        for k in range(self.mask.shape[-1]):
            cos = np.cos(np.deg2rad(phi))
            sin = np.sin(np.deg2rad(phi))
        
            Exx, Exy, Eyx, Eyy = E[:,:,k,0,0],E[:,:,k,0,1],E[:,:,k,1,0],E[:,:,k,1,1]
            Err[:,:,k] +=  cos*( cos*Exx+sin*Exy) +sin*( cos*Eyx+sin*Eyy)
            Ecc[:,:,k] += -sin*(-sin*Exx+cos*Exy) +cos*(-sin*Eyx+cos*Eyy)

        return Err, Ecc
    
   

def _roll(x, rx, ry):
    x = np.roll(x, rx, axis=0)
    return np.roll(x, ry, axis=1)

def _roll_to_center(x, cx, cy):
    nx, ny = x.shape[:2]
    return _roll(x,  int(nx//2-cx), int(ny//2-cy))

def _polar_grid(nx=128, ny=128):
    x, y = np.meshgrid(np.linspace(-nx//2, nx//2, nx), np.linspace(-ny//2, ny//2, ny))
    phi  = (np.rad2deg(np.arctan2(y, x)) + 180).T
    r    = np.sqrt(x**2+y**2+1e-8)
    return phi, r