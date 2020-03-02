import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_levelset(phi, level=0, img=None, colorbar=False):
    ax = plt.gca()
    if img is None:
        img = np.copy(phi)
    
    # plot image
    im = ax.imshow(img, interpolation='nearest')
    im.set_cmap('gray')
    _ = plt.axis('off')

    # plot contour
    contour = plt.contour(phi, [level], linewidths=2, colors = 'red')

    if colorbar:
        # position colorbar axis
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)

        # plot colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.add_lines(contour)
