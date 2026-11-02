import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.transforms as transforms
import numpy as np
import shapely
from shapely import convex_hull, MultiPoint
from shapely.plotting import plot_polygon

import datasets

# this is assuming 1 standard deviation
# adapted from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def cov_ellipse_vertices(x,y):
    cov = np.cov(x, y)
    pearson = cov[0,1] / np.sqrt(cov[0,0] * cov[1,1])
    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2)
    # Calculating the standard deviation of x, y from the square root of the variance
    # and multiplying with the given number of standard deviations.
    # n_std = 1 by assumption
    scale_x, mean_x = np.sqrt(cov[0,0]), x.mean()
    scale_y, mean_y = np.sqrt(cov[1,1]), y.mean()
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    return transf.transform_path(ellipse.get_path())._vertices


def path_cov_hull(paths):
    # shape is assumed to be T, N, 2
    assert paths.ndim == 3 and paths.shape[-1] == 2 
    T = len(paths)
    # get the vertices of the 1-std covariance ellipse at each timestep 
    ellipse_vertices = [cov_ellipse_vertices(paths[t,:,0], paths[t,:,1]) for t in range(T)]        
    geoms = [convex_hull(MultiPoint(
             np.vstack((ellipse_vertices[i], ellipse_vertices[i+1]))
             )) for i in range(T-1)]
    return shapely.unary_union(geoms)


def plot_cov_hull(paths, ax, color, **kwargs):
    plot_polygon(path_cov_hull(paths), ax=ax, facecolor=color,
                 add_points=False, linewidth=0, **kwargs)



def simple_cmaps(colors, end_color='black'):
    return [LinearSegmentedColormap.from_list(str(c), (c, end_color)) for c in colors]

def dataset_cmaps(dataset, end_color='black'):
    assert dataset in ['triangle', 'tmaze']
    return simple_cmaps(['C6', 'C1'] if dataset == 'tmaze' else \
                        ['C0', 'C6', 'C1', 'C3', 'C7', 'C2'])



def plot_reference_lines(ax, dataset, **kwargs):
    assert dataset in ['triangle', 'tmaze']
    if dataset == 'triangle':
        vertices, _ = datasets.triangle_vertices_and_mus()
        ax.plot(vertices[:2, 0], vertices[:2, 1], **kwargs)
        ax.plot(vertices[1:, 0], vertices[1:, 1], **kwargs)
        ax.plot(vertices[::2,0], vertices[::2,1], **kwargs)
    elif dataset == 'tmaze':
        ax.hlines(1, -1,1, **kwargs)
        ax.vlines(0,  0,1, **kwargs)
        


def add_border_and_ticks(ax, xlim, ylim, xticks=[0,1], yticks=[0,1],
                         linewidth=6, tickpercent=0.05, fill=None):
    ranges = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0]])
    ticklengths = ranges * tickpercent
    ax.set_xlim((xlim[0] - ticklengths[0], xlim[1]))
    ax.set_ylim((ylim[0] - ticklengths[1], ylim[1]))
    border = Rectangle((xlim[0], ylim[0]), ranges[0], ranges[1],
                       fill=(fill is not None), facecolor=fill, edgecolor='black',
                       linewidth=linewidth, clip_on=False)
    ax.add_artist(border)
    for xt in xticks:
        ax.vlines(xt, ylim[0]-ticklengths[0], ylim[0], color='k', linewidth=linewidth)
    for yt in yticks:
        ax.hlines(yt, xlim[0]-ticklengths[1], xlim[0], color='k', linewidth=linewidth)
        
        
        
def plot_triangle_power_diagram(ax, xlim, ylim, **kwargs):
    m = 1 / (2 * np.sin(np.pi/3))
    centroid = (0.5, np.tan(np.pi/6) / 2)
    # vertical line
    ax.plot([centroid[0]] * 2, [ylim[0], centroid[1]], **kwargs)
    # line from centroid to rightmost extremity
    ax.plot([centroid[0], xlim[1]], [centroid[1], xlim[1] * m], **kwargs)
    # line from centroid to leftmost extremity
    ax.plot([centroid[0], xlim[0]], [centroid[1], m * (1 - xlim[0])], **kwargs)
    
def plot_tmaze_power_diagram(ax, ylim, **kwargs):
    centroid = (0, 1)
    ax.vlines(-0.5, ylim[0], ylim[1], **kwargs)
    ax.vlines(0.5, ylim[0], ylim[1], **kwargs)