
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import find_contours
import numpy as np


def label_to_tags(label_3d, label_id=2):
    Ck = []
    for z in range(label_3d.shape[-1]):
        try:
            ck = tag_contour((label_3d[:,:,z].T==label_id)*1. )
            ck = np.concatenate((ck,np.zeros((24,3,1))+z),-1)
            Ck += [ck]
        except:
            continue
    return np.stack(Ck)

def label_to_tags2(contours_dict, K=24, R=5, centroid=None):
    Ck = []
    for z in contours_dict.keys():
        ck = tag_contour2(contours_dict[z], K=K, R=R, centroid=centroid)
        ck = np.concatenate((ck,np.zeros((K,R,1))+z),-1)
        Ck += [ck]
    return np.stack(Ck)

def tag_contour(myocardial_mask_2d, K=24, R=3):
    """Generate polar grid using myocardial mass mask.
    """    
    Cx, Cy = center_of_mass(myocardial_mask_2d)

    contours_endo, contours_epi  = find_contours(myocardial_mask_2d,0.8)

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
    for i in IDX:
        d = (x_endo[i]-x_epi)**2+(y_endo[i]-y_epi)**2
        dmin_i = np.argmin(d)

        rhos = np.linspace(rho_endo[i],rho_epi[dmin_i],R)
        xk   = [rhos[k] * np.cos(np.deg2rad(phi_endo[i]-180)) for k in range(R)] + Cx
        yk   = [rhos[k] * np.sin(np.deg2rad(phi_endo[i]-180)) for k in range(R)] + Cy
        ck  += [np.stack((yk,xk),-1)]

    ck = np.stack(ck,0)
    
    return ck

def tag_contour2(contours, phi_regionIDs, K=24, R=3, centroid=None):
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

        rhos = np.linspace(rho_endo[i],rho_epi[dmin_i],R)
        xk   = [rhos[k] * np.cos(np.deg2rad(phi_endo[i]-180)) for k in range(R)] + Cx
        yk   = [rhos[k] * np.sin(np.deg2rad(phi_endo[i]-180)) for k in range(R)] + Cy
        ck  += [np.stack((xk,yk),-1)]
        ck_regionIDs += [[phi_regionIDs[i]]*R]
        
    ck = np.stack(ck,0)
    ck_regionIDs = np.stack(ck_regionIDs,0)
    return ck, ck_regionIDs



def calculate_circumferential_strain(coords_batch, wall_index, use_linear_strain=False):
    # batch x time x 2 x 24
    midwall_points = coords_batch[:,:,:, wall_index::7]  # get point index 3 for every radial
    # print(midwall_points.shape)

    # we will have to calculate the strain between every points

    points_arr = np.split(midwall_points, 24, axis=3)

    # strain formula: ((l^2/L^2)-1) / 2  --> l^2 = x^2 + y^2
    # with x and y is the difference between x and y coords of 2 points
    ccs = []
    # the cc strain is circular, so we going through all of them and back to point 0
    for r in range(0,len(points_arr)):
        # for the last point, calculate between point_r and point_0
        if r+1 == len(points_arr):
            cc_diff = np.square(points_arr[r] - points_arr[0])
        else:
            cc_diff = np.square(points_arr[r] - points_arr[r+1])

        # do the sum: x^2 + y^2
        cc_sum = cc_diff[:,:,0] + cc_diff[:,:,1]

        if use_linear_strain:
            # use L instead of L^2
            cc_sum = np.sqrt(cc_sum)

        cc_sum_ed = cc_sum[:,0]

        # do the strain calculation
        partial_cc = cc_sum/cc_sum_ed[:, np.newaxis]
        if use_linear_strain:
            partial_cc = (partial_cc - 1)
        else:
            partial_cc = (partial_cc - 1) / 2

        # put the partial_cc in every time frame back together
        ccs.append(partial_cc)
    # stack the partial_cc for every links together
    stacked_ccs = np.stack(ccs, axis=2)

    # calculate the mean cc for every time frame
    mid_cc = np.mean(stacked_ccs, axis=2)
    # print(mid_cc.shape)
    # print(mid_cc[0][0:5])
    return mid_cc

def calculate_radial_strain(coords_batch, use_linear_strain=False):
    """
        Calculate rr strain for a batch of image sequences
        flattened_coords => [batch_size, nr_frames, 2, 168]
    """
    # point 0 is epi, point 6 is endo, do this for all the 'radials'
    endo_batch = coords_batch[:, :, :, ::7]
    epi_batch =  coords_batch[:, :, :, 6::7]

    # batch x time x 2 x 24 radials
    diff = (epi_batch - endo_batch) ** 2
    # print('diff', diff.shape)

    # batch x time x 24 sqrdiff
    summ = diff[:,:,0,:] + diff[:,:,1,:] # x^2 + y^2
    # print('summ', summ.shape)

    if use_linear_strain:
        # use L instead of L^2
        summ = np.sqrt(summ)

    # grab the frame 0 (ED) for all data, and 24 RR strains
    summ_ed = summ[:,0,:]

    # division through a certain column, without np.split
    # batch x time x 24 rr strains
    divv = summ/summ_ed[:,np.newaxis] # this is the trick, add new axis

    if use_linear_strain:
        rr_strains = divv - 1
    else:
        rr_strains = (divv - 1) / 2

    rr_strains = np.mean(rr_strains, axis=2)

    # batch x time x strain
    rr_strains = np.expand_dims(rr_strains, axis=2)
    return rr_strains