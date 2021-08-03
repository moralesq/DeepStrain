import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.measurements import center_of_mass

class Tensor():
    
    def __init__(self, Exx, Exy, Exz, Eyx, Eyy, Eyz, Ezx, Ezy, Ezz):
        
        self.E1, self.E2, self.E3 = Exx.copy(), Exy.copy(), Exz.copy()
        self.E4, self.E5, self.E6 = Eyx.copy(), Eyy.copy(), Eyz.copy()
        self.E7, self.E8, self.E9 = Ezx.copy(), Ezy.copy(), Ezz.copy()

    def asmat(self):
        return np.array([[self.E1,self.E2,self.E3],
                         [self.E4,self.E5,self.E6],
                         [self.E7,self.E8,self.E9]]).transpose((2,3,4,0,1))
        
    def asvoigt(self):
        return self.E1, self.E2, self.E3, self.E4, self.E5, self.E6, self.E7, self.E8, self.E9 
        
    def transpose(self):
        return Tensor(self.E1, self.E4, self.E7, self.E2, self.E5, self.E8, self.E3, self.E6, self.E9)
    
    def identity_add(self):
        self.E1 += 1; self.E5 += 1; self.E9 += 1 
    
    def identity_subtract(self):
        self.E1 -= 1; self.E5 -= 1; self.E9 -= 1 
        
    @staticmethod
    def dot(X, Y):
        
        X1,X2,X3,X4,X5,X6,X7,X8,X9=X.asvoigt()
        Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9=Y.asvoigt()
        
        Z1, Z2, Z3 = X1*Y1 + X2*Y4 + X3*Y7, X1*Y2 + X2*Y5 + X3*Y8, X1*Y3 + X2*Y6 + X3*Y9
        Z4, Z5, Z6 = X4*Y1 + X5*Y4 + X6*Y7, X4*Y2 + X5*Y5 + X6*Y8, X4*Y3 + X5*Y6 + X6*Y9
        Z7, Z8, Z9 = X7*Y1 + X8*Y4 + X9*Y7, X7*Y2 + X8*Y5 + X9*Y8, X7*Y3 + X8*Y6 + X9*Y9
        
        return Tensor(Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9)
    

class MyocardialStrain():
    
    def __init__(self, mask, flow):
                
        self.mask  = mask
        self.flow  = flow
        
        assert len(mask.shape) == 3
        assert len(flow.shape) == 4
        assert mask.shape == flow.shape[:3]
        assert flow.shape[-1] == 3

    def calculate_strain(self, lv_label=3):
        
        cx, cy, cz = center_of_mass(self.mask==lv_label)
        nx, ny, nz = self.mask.shape
        
        self.flow_rot = roll_to_center(self.flow, cx, cy)
        self.mask_rot = roll_to_center(self.mask, cx, cy)

        ux, uy, uz  = np.array_split(self.flow_rot, 3, -1)
        Uxx,Uxy,Uxz = np.gradient(np.squeeze(ux))
        Uyx,Uyy,Uyz = np.gradient(np.squeeze(uy))
        Uzx,Uzy,Uzz = np.gradient(np.squeeze(uz))
        
        F = Tensor(Uxx,Uxy,Uxz,Uyx,Uyy,Uyz,Uzx,Uzy,Uzz)

        F.identity_add()
        F = F.dot(F.transpose(), F) 
        F.identity_subtract()
        
        self.Err, self.Ecc, self.Erc, self.Ecr = convert_to_polar(mask=self.mask_rot, E=0.5*F.asmat()[:,:,:,:2,:2])
        

def roll(x, rx, ry):
    x = np.roll(x, rx, axis=0)
    return np.roll(x, ry, axis=1)

def roll_to_center(x, cx, cy):
    nx, ny = x.shape[:2]
    return roll(x,  int(nx//2-cx), int(ny//2-cy))

def polar_grid(nx=128, ny=128):
    x, y = np.meshgrid(np.linspace(-nx//2, nx//2, nx), np.linspace(-ny//2, ny//2, ny))
    phi  = (np.rad2deg(np.arctan2(y, x)) + 180).T
    r    = np.sqrt(x**2+y**2+1e-8)
    return phi, r

def convert_to_polar(mask, E):

    phi = polar_grid(*E.shape[:2])[0]
    Err = np.zeros(mask.shape)
    Ecc = np.zeros(mask.shape)
    Erc = np.zeros(mask.shape)
    Ecr = np.zeros(mask.shape)
    for k in range(mask.shape[-1]):
        cos = np.cos(np.deg2rad(phi))
        sin = np.sin(np.deg2rad(phi))

        Exx, Exy, Eyx, Eyy = E[:,:,k,0,0],E[:,:,k,0,1],E[:,:,k,1,0],E[:,:,k,1,1]
        Err[:,:,k] +=  cos*( cos*Exx+sin*Exy) + sin*( cos*Eyx+sin*Eyy)
        Ecc[:,:,k] += -sin*(-sin*Exx+cos*Exy) + cos*(-sin*Eyx+cos*Eyy)
        Erc[:,:,k] +=  cos*(-sin*Exx+cos*Exy) + sin*(-sin*Eyx+cos*Eyy)
        Ecr[:,:,k] += -sin*( cos*Exx+sin*Exy) + cos*( cos*Eyx+sin*Eyy)

    return Err, Ecc, Erc, Ecr

    
def convert_to_aha4d(tensor, mask):
    Tensor = tensor.copy()
    Mask   = mask.copy()

    Tensor[Mask!=2,:] = 0

    # rotate to have RV center of mass on the right
    angle  = _get_lv2rv_angle(Mask)
    Tensor = rotate(Tensor,  -angle, reshape=False, order=0)
    Mask   = rotate(Mask, -angle, reshape=False, order=1).clip(0,3).round()

    # roll to center 
    cx, cy = center_of_mass(Mask>1)[:2]
    Tensor = np.flipud(np.rot90(_roll_to_center(Tensor, cx, cy)))
    Mask   = np.flipud(np.rot90(_roll_to_center(Mask, cx, cy)))

    # remove slices that do not contain tissue labels
    ID     = Mask.sum(axis=(0,1))>0
    Mask   = Mask[:,:,ID]
    Tensor = Tensor[:,:,ID]
    
    return Tensor, Mask
    
class PolarMap():
    
    def __init__(self, Err, Ecc, mask):
        """ Plot Err and Ecc in a PolarMap using a segmentation of the heart as reference. 
            Assuming mask==2 yields the myocardium labels. 
        
        """
        
        self.Err  = Err
        self.Ecc  = Ecc
        self.mask = mask
        
    def project_to_aha_polar_map(self):
       
        Err  = self.Err.copy()
        Ecc  = self.Ecc.copy()
        mask = self.mask.copy()

        Err[mask!=2] = 0
        Ecc[mask!=2] = 0

        # rotate to have RV center of mass on the right
        angle = _get_lv2rv_angle(mask)
        Ecc   = rotate(Ecc,  -angle, reshape=False, order=0)
        Err   = rotate(Err,  -angle, reshape=False, order=0)
        mask  = rotate(mask, -angle, reshape=False, order=1).clip(0,3).round()
        
        # roll to center 
        cx, cy = center_of_mass(mask>1)[:2]
        Ecc  = np.flipud(np.rot90(_roll_to_center(Ecc, cx, cy)))
        Err  = np.flipud(np.rot90(_roll_to_center(Err, cx, cy)))
        mask = np.flipud(np.rot90(_roll_to_center(mask, cx, cy)))
        
        # remove slices that do not contain tissue labels
        ID   = mask.sum(axis=(0,1))>0
        mask = mask[:,:,ID]
        Err  = Err[:,:,ID]
        Ecc  = Ecc[:,:,ID]


        

        Err = Err.transpose((2,0,1))
        Ecc = Ecc.transpose((2,0,1))
        print('... radial strain')
        V_err = self._project_to_aha_polar_map(Err)
        print('... circumferential strain')
        V_ecc = self._project_to_aha_polar_map(Ecc)

        results = {'V_err':V_err, 'V_ecc':V_ecc, 'mask':mask}

        return results
        
    def _project_to_aha_polar_map(self, E, nphi=360, nrad=100, dphi=1):
        
        nz = E.shape[0]
        angles = np.arange(0, nphi, dphi)
        V      = np.zeros((nz, angles.shape[0], nrad))

        for rj in range(nz):

            Err_q  = _inpter2(E[rj])

            PHI, R = _polar_grid(*Err_q.shape)
            PHI = PHI.ravel()
            R   = R.ravel()

            for k, pmin in enumerate(angles):
                pmax = pmin + dphi/2.0

                # Get values for angle segment
                PHI_SEGMENT = (PHI>=pmin)&(PHI<=pmax)
                Rk   = R[PHI_SEGMENT]
                PHIk = PHI[PHI_SEGMENT]
                Vk   = Err_q.ravel()[PHI_SEGMENT]
                
                
                Rk = Rk[np.abs(Vk)!=0]
                Vk = Vk[np.abs(Vk)!=0]
                
                if len(Vk) == 0:
                    continue # this might not be the best
                Rk = _rescale_linear(Rk, rj, rj + 1)

                r = np.arange(rj, rj+1, 1.0/nrad)
                f = interp1d(Rk, Vk)
                v = f(r)

                V[rj,k] += v

        return V        
        
        
    def construct_polar_map(self, tensor, start=30, stop=70, sigma=12):

        E  = tensor.copy()
        mu = E[:,:,start:stop].mean()

        nz = E.shape[0]
        E  = np.concatenate(np.array_split(E[:,:,start:stop], nz), axis=-1)[0]

        old = E.shape[1]/nz*1.
        for j in range(nz-1):
            xi = int(old//2+j*old)
            xj = int(old+old//2+j*old)
            E[:,xi:xj] = gaussian_filter(E[:,xi:xj],sigma=sigma, mode='wrap')
            E[:,xi:xj] = gaussian_filter(E[:,xi:xj],sigma=sigma, mode='wrap')

        E = np.stack(np.array_split(E,nz,axis=1))

        E = gaussian_filter(E, sigma=sigma, mode='wrap')
        E = gaussian_filter(E, sigma=sigma, mode='wrap')

        E = [E[0][None]] + [E[1:3]] + np.array_split(E[3:], 2, axis=0)

        E = [np.mean(E[i], axis=0) for i in range(4)]
        E = np.concatenate(E, axis=1)

        old = E.shape[1]/4
        for j in range(3):
            xi = int(old//2+j*old)
            xj = int(old+old//2+j*old)
            E[:,xi:xj] = gaussian_filter(E[:,xi:xj], sigma=sigma, mode='wrap')
            E[:,xi:xj] = gaussian_filter(E[:,xi:xj], sigma=sigma, mode='wrap')

        E = gaussian_filter(E, sigma=sigma, mode='wrap')
        E = gaussian_filter(E, sigma=sigma, mode='wrap')

        mu = [mu] + self._get_17segments(E)

        return E, mu 
    
    def _get_17segments(self, data):
        c1,c2,c3,c4 = np.array_split(data,4,axis=-1)
        c2 = np.roll(c2,-45,0)
        #c2 = np.roll(c2,-90,0)

        c4 = [np.mean(ci) for ci in np.array_split(c4,6,axis=0)]
        c4 = list(np.roll(np.array(c4),-1))
        c3 = [np.mean(ci) for ci in np.array_split(c3,6,axis=0)]
        c3 = list(np.roll(np.array(c3),-1))
        c2 = [np.mean(ci) for ci in np.array_split(c2,4,axis=0)]
        #c2 = list(np.roll(np.array(c2),-1))
        c1 = [np.mean(c1)]

        c = c4 + c3 + c2 + c1

        return c

    def _get_17segments_RC(self, data1,data2):
        
        def _rc(a,b):
            #return np.mean(np.abs((b-a)/b)*100)
            return np.mean(((b-a)/b)*100)
        
        c1_1,c2_1,c3_1,c4_1 = np.array_split(data1,4,axis=-1)
        c1_2,c2_2,c3_2,c4_2 = np.array_split(data2,4,axis=-1)

        c4 = [_rc(ci1,ci2) for ci1,ci2 in zip(np.array_split(c4_1,6,axis=0),
                                              np.array_split(c4_2,6,axis=0))]
        c4 = list(np.roll(np.array(c4),-1))
        
        c3 = [_rc(ci1,ci2) for ci1,ci2 in zip(np.array_split(c3_1,6,axis=0),
                                              np.array_split(c3_2,6,axis=0))]
        c3 = list(np.roll(np.array(c3),-1))
        
        c2 = [_rc(ci1,ci2) for ci1,ci2 in zip(np.array_split(c2_1,4,axis=0),
                                              np.array_split(c2_2,4,axis=0))]
        c2 = list(np.roll(np.array(c2),-1))
        c1 = [_rc(c1_1,c1_2)]

        c = c4 + c3 + c2 + c1

        return c
    
def _roll(x, rx, ry):
    x = np.roll(x, rx, axis=0)
    return np.roll(x, ry, axis=1)

def _roll_to_center(x, cx, cy):
    nx, ny = x.shape[:2]
    return _roll(x,  int(nx//2-cx), int(ny//2-cy))

def _py_ang(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'. """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.rad2deg(np.arctan2(sinang, cosang))

def _polar_grid(nx=128, ny=128):
    x, y = np.meshgrid(np.linspace(-nx//2, nx//2, nx), np.linspace(-ny//2, ny//2, ny))
    phi  = (np.rad2deg(np.arctan2(y, x)) + 180).T
    r    = np.sqrt(x**2+y**2+1e-8)
    return phi, r

def _rescale_linear(array, new_min, new_max):
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max-new_min) / (maximum-minimum)
    b = new_min - m* minimum
    return m*array + b

def _inpter2(Eij, k=10):
    nx, ny = Eij.shape
    
    x  = np.linspace(0,nx-1,nx)
    y  = np.linspace(0,ny-1,ny)
    xq = np.linspace(0,nx-1,nx*k)
    yq = np.linspace(0,ny-1,ny*k)
    
    f = interp2d(x,y,Eij,kind='linear')
    
    return f(xq,yq)    
    
def _get_lv2rv_angle(mask):
    cx_lv, cy_lv = center_of_mass(mask>1)[:2]
    cx_rv, cy_rv = center_of_mass(mask==1)[:2]
    phi_angle    = _py_ang([cx_rv-cx_lv, cy_rv-cy_lv], [0, 1])
    return phi_angle    
    
    
    
### FUNCTIONS TO PLOT THE POLAR MAP 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

def _write(ax, mu, j, theta_i, i, width=2):
    xi, yi = polar2cart(0,  theta_i)
    xf, yf = polar2cart(35, theta_i)

    l = Line2D([40-xi,40-xf], [40-yi,40-yf], color='black', linewidth=width)
    ax.add_line(l)
    xi, yi = polar2cart(30, theta_i + 2*np.pi/12)
    ax.text(40-xi-.3, 40-yi, '%d' %(mu[j][i]), weight='bold', fontsize=14)
    
def write(ax, mu, j, width=2):
    if j > 1:
        for i in range(6):
            theta_i = 2*np.pi - i*60*np.pi/180 + 2*60*np.pi/180
            _write(ax, mu, j, theta_i, i)
    if j == 1:
        for i in range(4):
            theta_i = i*90*np.pi/180 - 45*np.pi/180
            _write(ax, mu, j, theta_i, i)
    if j ==0:
        ax.text(40-.3, 40, '%d' %(mu[j][0]), weight='bold', fontsize=14)

def plot_bullseye(data,mu,vmin=None,vmax=None, savepath=None,cmap='RdBu_r', label='GPRS (%)', 
                  std=None,cbar=False,color='white', fs=20, xshift=0, yshift=0, ptype='mesh',frac=False):
    
    rho     = np.arange(0,4,4.0/data.shape[1]);
    Theta   = np.deg2rad(range(data.shape[0]))
    [th, r] = np.meshgrid(Theta, rho);

    fig, ax = plt.subplots(figsize=(6,6))
    #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    #ax.axis('tight') creates errors 
    #ax.axis('off')
    if ptype == 'mesh':
        im = ax.pcolormesh(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    else:
        im = ax.contourf(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    if cbar:
        cbar = plt.colorbar(im, cax=fig.add_axes([0.15, -0.03, 0.7, 0.05]), orientation='horizontal')

        new_ticks = []
        new_ticks_labels = []
        for i,tick in enumerate(cbar.ax.get_xticks()):
            if i % 2 == 0:
                new_ticks.append(np.round(tick))
                new_ticks_labels.append(str(int(np.round(tick))))

        cbar.set_ticks(new_ticks);
        cbar.set_ticklabels(new_ticks_labels);

        # override if vmin is provided, assume vmax is provided too for now
        if vmin is not None:
            cbar.set_ticks([vmin, (vmax+vmin)/2.0, vmax]);
            cbar.set_ticklabels(['%d'%(i) for i in [vmin, (vmax+vmin)/2.0, vmax]]);
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(label, fontsize=26, weight='bold')

    ax.axis('off')
    if std is not None:
        draw_circle_group(ax,100*np.array(mu),100*np.array(std))
    if frac:
        draw_circle_frac(ax,np.array(mu), color=color)
    else:
        draw_circle(ax,100*np.array(mu), color=color, fs=fs, xshift=xshift, yshift=yshift)
    if savepath is not None:
        if not cbar:
            plt.tight_layout()
        plt.savefig(savepath, dpi=600)
    plt.show()
    
def plot_bullseye_ratio(data,mu,vmin=None,vmax=None, savepath=None,cmap='RdBu_r', label='GPRS (%)', 
                  std=None,cbar=False,color='white',ptype='mesh',frac=False):
    
    rho     = np.arange(0,4,4.0/data.shape[1]);
    Theta   = np.deg2rad(range(data.shape[0]))
    [th, r] = np.meshgrid(Theta, rho);

    fig, ax = plt.subplots(figsize=(6,6))

    if ptype == 'mesh':
        im = ax.pcolormesh(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    else:
        im = ax.contourf(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,shading='gouraud')
    cbar = plt.colorbar(im, cax=fig.add_axes([0.15, -0.03, 0.7, 0.05]), orientation='horizontal')

    draw_circle_error(ax)
    ax.axis('off')
    if savepath is not None:
        if not cbar:
            plt.tight_layout()
        plt.savefig(savepath, dpi=600)
    plt.show()
    
def plot_bullseye_error(data,mu,vmin=None,vmax=None, savepath=None,cmap='RdBu_r', label='GPRS (%)',n=5):
    
    rho     = np.arange(0,4,4.0/data.shape[1]);
    Theta   = np.deg2rad(range(data.shape[0]))
    [th, r] = np.meshgrid(Theta, rho);

    fig, ax = plt.subplots(figsize=(6,6))

    levels = np.linspace(vmin, vmax, n+1)
    im = ax.contourf(r*np.cos(Theta), r*np.sin(Theta), 100*data.T, 
                           vmin=vmin,vmax=vmax,cmap=cmap,levels=levels)

    cbar = plt.colorbar(im, cax=fig.add_axes([0.15, -0.03, 0.7, 0.05]), orientation='horizontal')

    #ticks = -np.array(range(0,120,20))

    #cbar.set_ticks(ticks);
    #cbar.set_ticklabels(['%d'%(i) for i in ticks]);

     

    ax.axis('off')
    draw_circle_error(ax)
    if savepath is not None:
        if not cbar:
            plt.tight_layout()
        plt.savefig(savepath, dpi=500)
    plt.show()

def draw_circle_error(ax,width=4):

    circle1 = plt.Circle((0,0), 1, color='black', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='black', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='black', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='black', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    j = 0
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)

    j += 6
    for i in range(4):
        theta_i = i*90*(np.pi/180) - 45
        xi, yi  = polar2cart(1, theta_i)
        xf, yf  = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)

def draw_circle_frac(ax, mu, width=4, fs=20, color='white'):
    
    
    
    circle1 = plt.Circle((0,0), 1, color='black', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='black', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='black', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='black', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    j = 0
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(3.5, theta_i + 2*np.pi/12)
        ax.text(xi-.3, yi, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color);
        xi, yi = polar2cart(2.5, theta_i + 2*np.pi/12)
        ax.text(xi-.3, yi, '%.2f' %(mu[j+6]), weight='bold', fontsize=fs, color=color); j += 1
        
    j += 6
    LABELS = ['ANT', 'SEPT', 'INF', 'LAT']
    for i in range(4):
        theta_i = i*90*np.pi/180 - 45 
        xi, yi = polar2cart(1, theta_i)
        xf, yf = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(1.5, theta_i + 2*np.pi/8)

        ax.text(xi-.3, yi, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color); j += 1;
        xi, yi = polar2cart(5, theta_i + 2*np.pi/8)

    ax.text(0-.3, 0-.3, '%.2f' %(mu[j]), weight='bold', fontsize=fs, color=color)
    
def draw_circle(ax, mu, width=4, fs=15, xshift=0, yshift=0, color='white'):
    
    
    
    circle1 = plt.Circle((0,0), 1, color='black', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='black', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='black', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='black', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    j = 0
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(3.5, theta_i + 2*np.pi/12)
        ax.text(xi-.4-xshift, yi-yshift, '%d' %(mu[j]), weight='bold', fontsize=fs, color=color);
        xi, yi = polar2cart(2.5, theta_i + 2*np.pi/12)
        ax.text(xi-.4-xshift, yi-yshift, '%d' %(mu[j+6]), weight='bold', fontsize=fs, color=color); j += 1
        
    j += 6
    LABELS = ['ANT', 'SEPT', 'INF', 'LAT']
    for i in range(4):
        theta_i = i*90*np.pi/180  + 45*np.pi/180
        xi, yi = polar2cart(1, theta_i)
        xf, yf = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(1.5, theta_i + 2*np.pi/8)

        ax.text(xi-.4-xshift, yi-yshift, '%d' %(mu[j]), weight='bold', fontsize=fs, color=color); j += 1;
        xi, yi = polar2cart(5, theta_i + 2*np.pi/8)

    ax.text(-.4-xshift, 0-yshift, '%d' %(mu[j]), weight='bold', fontsize=fs, color=color)
  

def draw_circle_group(ax, mu, std, width=4, fs=14, color='white'):
    
        
    circle1 = plt.Circle((0,0), 1, color='black', fill=False, linewidth=width)
    circle2 = plt.Circle((0,0), 2, color='black', fill=False, linewidth=width)
    circle3 = plt.Circle((0,0), 3, color='black', fill=False, linewidth=width)
    circle4 = plt.Circle((0,0), 4, color='black', fill=False, linewidth=width)

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    
    j = 0
    for i in range(6):
        theta_i = i*60*np.pi/180 + 60*np.pi/180
        xi, yi = polar2cart(2, theta_i)
        xf, yf = polar2cart(4, theta_i)
        
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(3.5, theta_i + 2*np.pi/12)
        ax.text(xi-.6, yi, '%d(%d)' %(mu[j],std[j]), weight='bold', fontsize=fs, color=color);
        xi, yi = polar2cart(2.5, theta_i + 2*np.pi/12)
        ax.text(xi-.6, yi, '%d(%d)' %(mu[j+6],std[j+6]), weight='bold', fontsize=fs, color=color); j += 1
        
    j += 6
    LABELS = ['ANT', 'SEPT', 'INF', 'LAT']
    for i in range(4):
        theta_i = i*90*np.pi/180 
        xi, yi = polar2cart(1, theta_i)
        xf, yf = polar2cart(2, theta_i)
        l = Line2D([xi,xf], [yi,yf], color='black', linewidth=width)
        ax.add_line(l)
        
        xi, yi = polar2cart(1.5, theta_i + 2*np.pi/8)

        ax.text(xi-.6, yi-0.1, '%d(%d)' %(mu[j],std[j]), weight='bold', fontsize=fs, color=color); j += 1;
        xi, yi = polar2cart(5, theta_i + 2*np.pi/8)

    ax.text(0-.3, 0-.2, '%d(%d)' %(mu[j],std[j]), weight='bold', fontsize=fs, color=color)
    
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y