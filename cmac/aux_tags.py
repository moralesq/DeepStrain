import numpy as np
from scipy import interpolate 


def fit_bsplice2d(phi_sorted, x, N=100, s=8, k=4):

    phi_close = np.array([0]+list(phi_sorted)+[360])
    x_close   = np.array([(x[0]+x[-1])/2]+list(x)+[(x[0]+x[-1])/2])
    t, c, k = interpolate.splrep(phi_close, x_close, s=s, k=k)
    spline  = interpolate.BSpline(t, c, k, extrapolate=False)
    return spline(np.linspace(0,360,N))

def fit_bsplice3d(phi_sorted, x, y, N=100, with_spline=False, s=8, k=4):
    if with_spline:
        xx = fit_bsplice2d(phi_sorted, x, N=N, s=s, k=k)
        yy = fit_bsplice2d(phi_sorted, y, N=N, s=s, k=k)
    else:
        xx = np.interp(np.linspace(0,360,N), phi_sorted, x, period=360)
        yy = np.interp(np.linspace(0,360,N), phi_sorted, y, period=360)
    return xx, yy

def interpolate_contour(x, y, regionIDs, subpartIDs, regionID_dic, subpartID_dic, centroid=None, N=100, s=8, k=4):
    x_endo, y_endo = x[subpartIDs==subpartID_dic['endocardium']], y[subpartIDs==subpartID_dic['endocardium']]
    x_epi, y_epi   = x[subpartIDs==subpartID_dic['epicardium']], y[subpartIDs==subpartID_dic['epicardium']]
    label_endo     = regionIDs[subpartIDs==subpartID_dic['endocardium']]
    label_epi      = regionIDs[subpartIDs==subpartID_dic['epicardium']]
        
    if centroid is None:
        Cx, Cy = (x_endo.max()+x_endo.min())/2, (y_endo.max()+y_endo.min())/2
    else:
        Cx, Cy = centroid
    
    phi_endo = np.rad2deg(np.arctan2(y_endo-Cy, x_endo-Cx))+180
    phi_epi  = np.rad2deg(np.arctan2(y_epi-Cy, x_epi-Cx))+180

    # start clockwise angles at the end of septal wall
    phi_endo -= phi_epi[(label_epi!=regionID_dic['septal'])&(label_epi!=regionID_dic['lateral'])].min()
    phi_epi  -= phi_epi[(label_epi!=regionID_dic['septal'])&(label_epi!=regionID_dic['lateral'])].min()
    phi_epi[phi_epi<0]   += 360
    phi_endo[phi_endo<0] += 360
    
    endo_idx = np.argsort(phi_endo)
    epi_idx  = np.argsort(phi_epi)
    
    phi_septal  = phi_epi[label_epi==regionID_dic['septal']]
    phi_lateral = phi_epi[label_epi==regionID_dic['lateral']]
    
    x_endo_spline, y_endo_spline = fit_bsplice3d(phi_endo[endo_idx],
                                                 x_endo[endo_idx],y_endo[endo_idx],N=N,s=s,k=k)
    x_epi_spline, y_epi_spline   = fit_bsplice3d(phi_epi[epi_idx],
                                                 x_epi[epi_idx],y_epi[epi_idx],N=N,s=s,k=k)
    
    phi_coords = np.linspace(0,360,N)
    phi_regionIDs = np.zeros(phi_coords.shape)
    phi_regionIDs[(phi_coords>phi_septal.min())&(phi_coords<phi_septal.max())]   = regionID_dic['septal']
    phi_regionIDs[(phi_coords>phi_lateral.min())&(phi_coords<phi_lateral.max())] = regionID_dic['lateral']
    
    return x_endo_spline, y_endo_spline, x_epi_spline, y_epi_spline,  phi_regionIDs

def tag_contour3d(contours_dict, regionIDs_dict, K=24, R=5, centroid=None):
    Ck = []
    Ck_rID = []
    for z in contours_dict.keys():
        ck, ck_rID = tag_contour2d(contours_dict[z], regionIDs_dict[z], K=K, R=R, centroid=centroid)
        ck = np.concatenate((ck,np.zeros((K,R,1))+z),-1)
        Ck += [ck]
        Ck_rID += [ck_rID]
    return np.stack(Ck), np.stack(Ck_rID)

def tag_contour2d(contours, regionIDs, K=24, R=3, centroid=None):
    """Generate polar grid using endocardial and epicardial contours.
    """
    get_cx_cy = lambda contour : (contour.max(axis=0) + contour.min(axis=0)) / 2
    
    contours_endo, contours_epi  = contours['endocardium'], contours['epicardium']
    
    if centroid is None:
        Cx, Cy = get_cx_cy(contours_endo)
    else: 
        Cx, Cy = centroid
    
    # cartesian coordinates (centered) of endo/epi contours
    x_endo, y_endo = contours_endo[:,0], contours_endo[:,1]
    x_epi, y_epi   = contours_epi[:,0], contours_epi[:,1]
    
    # polar coordinates of endo/epi contours
    phi_endo = np.rad2deg(np.arctan2(y_endo-Cy, x_endo-Cx))+180
    phi_epi  = np.rad2deg(np.arctan2(y_epi-Cy, x_epi-Cx))+180
    rho_endo = ((x_endo-Cx)**2 + (y_endo-Cy)**2)**0.5
    rho_epi  = ((x_epi-Cx)**2 + (y_epi-Cy)**2)**0.5

    IDX = [np.array_split(np.argsort(phi_endo),K)[k][0] for k in range(K)]
    
    
    ck = []
    ck_regionIDs = []
    for i in IDX:
        
        d = (x_endo[i]-x_epi)**2+(y_endo[i]-y_epi)**2
        dmin_i = np.argmin(d)

        rhos = np.linspace(rho_endo[i]*0.99,rho_epi[dmin_i]*1.01,R)
        xk   = [rhos[k] * np.cos(np.deg2rad(phi_endo[i]-180)) for k in range(R)] + Cx
        yk   = [rhos[k] * np.sin(np.deg2rad(phi_endo[i]-180)) for k in range(R)] + Cy
        ck  += [np.stack((xk,yk),-1)]
        ck_regionIDs += [[regionIDs[i]]*R]

    return np.stack(ck,0), np.stack(ck_regionIDs,0)