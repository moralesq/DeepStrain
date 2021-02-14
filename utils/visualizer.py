# Manuel A. Morales (moralesq@mit.edu)
# Harvard-MIT Department of Health Sciences & Technology  
# Athinoula A. Martinos Center for Biomedical Imaging

import numpy as np
from skimage import measure
import matplotlib.pylab as plt
from scipy.ndimage.morphology import binary_fill_holes

def Contours(mask, tissue_labels=[1,2,3]):
    contours = []
    for i in tissue_labels:
        mask_ = binary_fill_holes(mask==i)
        c = measure.find_contours(mask_,0.8)
        c = c[np.argmax([len(c) for c in c])]
        contours.append(c)
    return contours

def PlotContours(ax, mask, tissue_labels=[1,2,3], 
                 contour_colors=['lime','magenta','red'],
                 contour_labels=['RV','LVM','LV'],
                 tolerance=0.1,
                 alpha=1,
                 linewidth=2,
                 legend=False):
    
    contours = Contours(mask, tissue_labels=tissue_labels)
    for i, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0],alpha=alpha, linewidth=linewidth, color=contour_colors[i],label=contour_labels[i])
    if legend:
        ax.legend(fontsize=26)  
        
        
def Plot(image, mask=None, figsize=(7,7), crop=False):
    
    if crop:
        nx, ny = image.shape
        image = image[nx//2-64:nx//2+64,ny//2-64:ny//2+64]
        if mask is not None:
            mask = mask[nx//2-64:nx//2+64,ny//2-64:ny//2+64]
    
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.imshow(image, cmap='gray')
    if mask is not None:
        try:
            PlotContours(ax, mask, alpha=1, linewidth=1.3)  
        except:
            print('No contours found!')
            

            
            
